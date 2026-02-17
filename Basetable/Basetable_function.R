#' ' Generate a Baseline Table for AMP-SCZ or Prescient Data
#'
#' This function processes raw AMP-SCZ or Prescient study data, extracts relevant variables,
#' and compiles them into a structured baseline table. It automatically detects the data structure
#' (AMP-SCZ vs. Prescient), performs necessary calculations, and saves the final dataset as a CSV file.
#'
#' @param DIR A string specifying the directory where the raw data is stored.
#'            The function will scan this directory to determine the dataset type.
#' @param dictionary_DIR A directory to where the dictionary is saved.
#' @param vars A character vector containing the variable names that should be extracted from the data.
#'
#' @return A data frame (`T`) containing cleaned and structured baseline data.
#'         The processed table includes the selected variables, a `group` variable (where 1 = CHR and 0 = HC),
#'         and any computed scores specific to the dataset type.
#'
#' @details
#' **1. Automatic Data Structure Detection:**
#' - If the directory contains filenames with date patterns (DD.MM.YYYY), it assumes the data is from the **Prescient study**.
#' - If the directory contains subfolders with numeric identifiers (e.g., `/csv/65283/`), it assumes the data is from **AMP-SCZ**.
#' - The function dynamically adjusts its processing steps based on the detected structure.
#'
#' **2. Data Processing Steps:**
#' - For **Prescient data**, the function:
#'   - Extracts relevant variables from the latest dataset version.
#'   - Calculates total scores for CDSS, NSIPR, OASIS, and CHR duration.
#'   - Filters out test cases and missing values.
#'   - Standardizes and cleans variable formats (e.g., age conversion, recoding of categorical variables).
#'   - Merges data from different sources into a unified table.
#'
#' - For **AMP-SCZ data**, the function:
#'   - Identifies the latest dataset version automatically.
#'   - Extracts subject-level variables from multiple files.
#'   - Converts certain categorical variables into factors.
#'   - Merges data from different sources into a unified table.
#'
#' **3. Handling Missing and Ineligible Data:**
#' - Removes records with entirely missing variable values.
#' - Excludes test cases and ineligible subjects.
#' - Maps subject IDs to site locations for better interpretability.
#'
#' **4. Output:**
#' - The cleaned and processed dataset is saved in a subdirectory `"BASELINE_TABLES"` under `data_DIR`.
#' - The output filename is formatted as `"basetable_YYYY-MM-DD.csv"`, where `YYYY-MM-DD` represents the current date.
#' - The final table contains:
#'   - The requested variables (`vars`).
#'   - A unified subject ID column (`subjectkey` or `src_subject_id`).
#'   - A standardized `group` column (`1 = CHR`, `0 = HC`).
#'   - Any computed or transformed variables.
#'
#' @import dplyr readr stringr tidyr
#' @export
#'
#' @examples
#' # Example usage for Prescient data
#' data_DIR <- "/path/to/data/Prescient_01.10.2024"
#' dictionary_DIR <- "~/Library/CloudStorage/OneDrive-Personal/OneDrive/PhD computational mental health/Code/GIT/Docs/1Complete_dictionary.xlsx"

#' baseline_table <- create_basetable(data_DIR, vars = vars)
#'
#' # Example usage for AMP-SCZ data
#' data_DIR <- "/path/to/data/3705"
#' data <- create_basetable( data_DIR = data_DIR, dictionary_DIR = dictionary_DIR)
#' 

#' ------------------------------------------------------------------------------------
# Generate a Baseline Table for AMP-SCZ or Prescient Data
# (Optimized version with the exact same workflow)
#' ------------------------------------------------------------------------------------

create_basetable <- function(data_DIR, dictionary_DIR, release = NULL, timepoint= "baseline") {
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # 0) Load libraries -------
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(forcats)
  library(readxl)
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # 1) Define sub-functions ------
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  
  variable_selection <- function(dictionary_DIR, required_vars = c("phenotype", "group")) { 
    dict <- read_excel(dictionary_DIR, 
                       col_types = "text")
    
    # Ensure required columns exist
    if (!"Include_basetable" %in% names(dict)) {
      stop("Error: Dictionary must have an 'Include_basetable' column.")
    }
    if (!"Include_demographics" %in% names(dict)) {
      stop("Error: Dictionary must have an 'Include_demographics' column.")
    }
    if (!"Datatype" %in% names(dict)) {
      stop("Error: Dictionary must have a 'Datatype' column if you want to track data sources.")
    }
    
    
    # Expand Aliases into separate rows
    dict <- dict %>%
      mutate(Aliases = str_replace_all(Aliases, "\\s+", "")) %>%
      separate_rows(Aliases, sep = ",") %>%
      mutate(Aliases = ifelse(Aliases == "", NA, Aliases))
    
    # Create duplicated rows for each alias
    alias_rows <- dict %>%
      filter(!is.na(Aliases)) %>%
      mutate(
        Original_ElementName = ElementName,
        ElementName = Aliases
      ) %>%
      select(-Aliases)
    
    # Combine with the original dictionary
    dict <- bind_rows(dict, alias_rows) %>%
      select(-Aliases, -Original_ElementName, -Condition)
    
    # Select only variables for the base table
    selected_vars <- dict %>%
      filter(Include_basetable %in% c("Yes", "yes")) %>%
      select(ElementName, Label)
    
    # Select variables for the demographics
    selected_vars_demographics <- dict %>%
      filter(Include_demographics %in% c("Yes", "yes")) %>%
      select(ElementName, Label)
    
    # Create metatable with info about modality
    metatable <- dict %>%
      filter(Include_basetable %in% c("Yes", "yes")) %>%
      select(ElementName, Datatype, Modality)
    
    # Ensure required_vars are included
    missing_required <- setdiff(required_vars, selected_vars$ElementName)
    if (length(missing_required) > 0) {
      selected_vars <- bind_rows(
        selected_vars,
        tibble(ElementName = missing_required, Label = missing_required)
      )
    }
    
    selected_vars <- unique(selected_vars)
    
    # Extract final sets
    vars_basetable    <- selected_vars$ElementName
    vars_demographics <- selected_vars_demographics$ElementName
    labels            <- setNames(as.list(selected_vars$Label), selected_vars$ElementName)
    
    return(list(vars_basetable, vars_demographics, labels, metatable))
  }
  
  #' ----------------------------------------------------------------------------------
recode_if_present <- function(.data, colname) {
  
  # 1) Read & expand dictionary
  dict_raw <- suppressMessages(
    readxl::read_excel(dictionary_DIR, col_types = "text")
  )
  dict_expanded <- dict_raw %>%
    mutate(Aliases = str_remove_all(Aliases, "\\s+")) %>%
    tidyr::separate_rows(Aliases, sep = ",") %>%
    mutate(lookup_name = if_else(is.na(Aliases) | Aliases == "",
                                 ElementName,
                                 Aliases)) %>%
    select(lookup_name, Notes) %>%
    distinct(lookup_name, .keep_all = TRUE)
  
  
  
  # Only act if the column exists in the data
  if (colname %in% names(.data)) {
    
    # Change the calgory depression columns so recoding is according to dictionary
    calg_names <- c(
      'chrcdss_calg1', 'chrcdss_calg2', 'chrcdss_calg3',
      'chrcdss_calg4', 'chrcdss_calg5', 'chrcdss_calg6',
      'chrcdss_calg7', 'chrcdss_calg8', 'chrcdss_calg9'
    )
    
    if (colname %in% calg_names) {
      # Find all entries that are exactly 0, 1, 2 or 3
      idx <- which(.data[[colname]] %in% 0:3)
      # Add 1 to those positions
      .data[[colname]][idx] <- .data[[colname]][idx] + 1
    }
    
    # Retrieve the recoding note for this variable from the dictionary.
    note <- dict_expanded$Notes[dict_expanded$lookup_name == colname]
    
    # If no note is found or it's empty, skip recoding.
    if (length(note) == 0 || is.na(note) || note == "") {
      return(.data)
    }
    
    # If there is conditional logic in the note (e.g., "if"), only use the part before it.
    # If the note contains "-900", strip "-900" and everything after it:
    note_clean <- if (grepl("-900", note, fixed = TRUE)) {
      sub("-900.*$", "", note)
    } else {
      note
    }
    note_clean <- trimws(note_clean)
    # Remove any trailing semicolons
    note_clean <- sub("[;]+$", "", note_clean)
    
    # Split the note into potential recode pairs by semicolons.
    recode_pairs <- unlist(strsplit(note_clean, "[;]"))
    recode_pairs <- trimws(recode_pairs)
    # Keep only valid pairs that contain "="
    recode_pairs <- recode_pairs[grepl("=", recode_pairs)]
    
    # Only proceed if there are at least two recode pairs (a clear mapping).
    if (length(recode_pairs) < 2) {
      return(.data)
    }
    
    # Extract original keys and new values from each recode pair.
    orig_keys <- sapply(recode_pairs, function(pair) {
      parts <- unlist(strsplit(pair, "="))
      trimws(parts[1])
    })
    new_values <- sapply(recode_pairs, function(pair) {
      parts <- unlist(strsplit(pair, "="))
      if (length(parts) >= 2) {
        trimws(parts[2])
      } else {
        NA
      }
    })
    names(new_values) <- orig_keys
    
    # If the only recode pairs are for missing values (-900 and -300), do not recode.
    if (all(orig_keys %in% c("-900", "-300", "-7", "-998", "-997", "-9", "-3"))) {
      return(.data)
    }
    
    # Determine if the recode mapping is complete, ignoring NA values.
    unique_vals <- as.character(unique(.data[[colname]]))
    unique_vals <- unique_vals[!is.na(unique_vals) & !(unique_vals %in% c("-300", "-900", "N/A", "<NA>", "NA", -300, -7, "-7", "-997","-998", "-9", "-3"))]
    
    if (all(unique_vals %in% orig_keys)) {
      # Full mapping: recode normally, so "1" becomes "least" if mapping is 1=least.
      recode_mapping <- setNames(orig_keys, new_values)
      
    } else {
      # Partial mapping: create composite labels for mapped values (e.g., "1=least")
      composite_labels <- paste0(orig_keys, "=", new_values)
      recode_mapping <- setNames(orig_keys, composite_labels)
      print(paste("Partial recoding for:", colname))
    }
    
    # Convert the column to a factor and apply the recoding.
    .data[[colname]] <- as.factor(.data[[colname]])
    .data <- .data %>%
      dplyr::mutate(!!colname := suppressWarnings(forcats::fct_recode(.data[[colname]], !!!recode_mapping)))
  }
  
  return(.data)
}
  

  
  #' ----------------------------------------------------------------------------------
  # calc_total.R
  #
  # R function to calculate a row-wise total from a matrix or data frame of numeric values.
  # If an entire row is NA, the result is NA for that row; otherwise, sum the row ignoring NA.
  
  calc_total <- function(data) {
    
    # Make sure data is either a matrix or data frame
    if (!is.matrix(data) && !is.data.frame(data)) {
      stop("Input 'data' must be a matrix or data frame.")
    }
    
    # If data is a data frame, convert to matrix (assuming all numeric)
    if (is.data.frame(data)) {
      data <- as.matrix(data)
    }
    
    # Calculate row totals
    # If the row is all NA, return NA, otherwise sum ignoring NA
    total <- apply(data, 1, function(row_vals) {
      if (all(is.na(row_vals))) {
        return(NA)
      } else {
        return(sum(row_vals, na.rm = TRUE))
      }
    })
    
    return(total)
  }
  
  
  #' ----------------------------------------------------------------------------------
  # manage_missing.R
  #
  # R function to replace given missing values with NA in a data frame or matrix,
  
  
  manage_missing <- function(T, missing_values) {
    
    # Check if T is a data frame:
    if (is.data.frame(T)) {
      # Loop over columns
      for (i in seq_along(T)) {
        col <- T[[i]]
        
        # Skip columns that are character, logical, or date/time
        # (check for either Date or POSIX date/time classes)
        if (!is.character(col) &&
            !is.logical(col)   &&
            !inherits(col, "Date") &&
            !inherits(col, "POSIXt")) {
          
          # Replace matching missing_values with NA
          T[[i]][ T[[i]] %in% missing_values ] <- NA
        }
      }
      
      # Alternatively, if T is a matrix, apply the same logic column-wise
    } else if (is.matrix(T)) {
      # For each column
      for (i in seq_len(ncol(T))) {
        # We assume this is numeric or something comparable
        col_values <- T[, i]
        # Replace matching missing_values with NA
        col_values[col_values %in% missing_values] <- NA
        T[, i] <- col_values
      }
      
    } else {
      warning("Input T is neither a data frame nor a matrix. Returning unchanged.")
    }
    
    return(T)
  }
  
  
  
  
  #' ----------------------------------------------------------------------------------
  # calculate_chr_duration.R
  #
  # Reads multiple CSV files containing subjectkey, interview_date, and onset variables
  # and performs calculations to derive CHR durations, APS durations, CAARMS durations, etc.
  
  # If you use lubridate, uncomment the library call:
  # install.packages("lubridate")
  # library(lubridate)
  
  calculate_chr_duration <- function(fileNames) {
    
    # Define sets of columns (onset and filter columns) to extract
    onset_vars <- c(
      "chrpsychs_scr_1a1_on3", "chrpsychs_scr_2a1_on3", "chrpsychs_scr_3a1_on3",
      "chrpsychs_scr_4a1_on3", "chrpsychs_scr_5a1_on3", "chrpsychs_scr_6a1_on3",
      "chrpsychs_scr_7a1_on3", "chrpsychs_scr_8a1_on3", "chrpsychs_scr_9a1_on3",
      "chrpsychs_scr_10a1_on3","chrpsychs_scr_11a1_on3","chrpsychs_scr_12a1_on3",
      "chrpsychs_scr_13a1_on3","chrpsychs_scr_14a1_on3","chrpsychs_scr_15a1_on3"
    )
    sips_aps_lifetime <- c(
      "chrpsychs_scr_1a14","chrpsychs_scr_2a14","chrpsychs_scr_3a14",
      "chrpsychs_scr_4a14","chrpsychs_scr_5a14","chrpsychs_scr_6a14",
      "chrpsychs_scr_7a14","chrpsychs_scr_8a14","chrpsychs_scr_9a14",
      "chrpsychs_scr_10a14","chrpsychs_scr_11a14","chrpsychs_scr_12a14",
      "chrpsychs_scr_13a14","chrpsychs_scr_14a14","chrpsychs_scr_15a14"
    )
    caarms_pastyear <- c(
      "chrpsychs_scr_1b19","chrpsychs_scr_2b19","chrpsychs_scr_3b19",
      "chrpsychs_scr_4b19","chrpsychs_scr_5b19","chrpsychs_scr_6b19",
      "chrpsychs_scr_7b19","chrpsychs_scr_8b19","chrpsychs_scr_9b19",
      "chrpsychs_scr_10b19","chrpsychs_scr_11b19","chrpsychs_scr_12b19",
      "chrpsychs_scr_13b19","chrpsychs_scr_14b19","chrpsychs_scr_15b19"
    )
    sips_aps_progression <- c(
      "chrpsychs_scr_1d22","chrpsychs_scr_2d22","chrpsychs_scr_3d22",
      "chrpsychs_scr_4d22","chrpsychs_scr_5d22","chrpsychs_scr_6d22",
      "chrpsychs_scr_7d22","chrpsychs_scr_8d22","chrpsychs_scr_9d22",
      "chrpsychs_scr_10d22","chrpsychs_scr_11d22","chrpsychs_scr_12d22",
      "chrpsychs_scr_13d22","chrpsychs_scr_14d22","chrpsychs_scr_15d22"
    )
    
    all_variables <- c("subjectkey", "interview_date",
                       onset_vars,
                       sips_aps_lifetime,
                       caarms_pastyear,
                       sips_aps_progression)
    
    # We'll accumulate into this data frame
    finalTable <- NULL
    
    # Helper to do a safe "outer join" by subjectkey
    # We also carefully handle interview_date columns 
    do_full_join <- function(df1, df2) {
      # full_join by subjectkey
      # suffixes to distinguish new columns
      joined <- dplyr::full_join(df1, df2, by = "subjectkey", suffix = c(".x", ".y"))
      
      # Merge interview_date columns if both exist
      if ("interview_date.x" %in% names(joined) && 
          "interview_date.y" %in% names(joined)) {
        
        # Try coalescing
        joined$interview_date <- dplyr::coalesce(joined$interview_date.x,
                                                 joined$interview_date.y)
        # Remove old columns
        joined$interview_date.x <- NULL
        joined$interview_date.y <- NULL
        
      } else if ("interview_date" %in% names(joined)) {
        # Already a single interview_date column
        # Nothing to do
      } else if ("interview_date.x" %in% names(joined)) {
        # Maybe only the .x version exists
        joined$interview_date <- joined$interview_date.x
        joined$interview_date.x <- NULL
      } else if ("interview_date.y" %in% names(joined)) {
        # Maybe only the .y version exists
        joined$interview_date <- joined$interview_date.y
        joined$interview_date.y <- NULL
      }
      return(joined)
    }
    
    # 1) Read each CSV in fileNames and outer-join it into finalTable
    for (filePath in fileNames) {
      tempTable <- tryCatch(
        {
          df <- suppressWarnings(read_csv(filePath, show_col_types = FALSE))
          
          
          # Keep only the intersecting columns
          commonVars <- intersect(names(df), all_variables)
          # Must include at least "subjectkey"
          if (!"subjectkey" %in% commonVars) {
            stop(paste("'subjectkey' is missing in", filePath))
          }
          df <- df[, commonVars, drop = FALSE]
          
          # Return
          df
        },
        error = function(e) {
          stop(paste("Error reading file:", filePath, " -> ", e$message))
        }
      )
      
      if (is.null(finalTable)) {
        # First file becomes finalTable
        finalTable <- tempTable
      } else {
        # Full (outer) join
        finalTable <- do_full_join(finalTable, tempTable)
      }
    }
    
    # Make sure interview_date is in Date or POSIXct format for date arithmetic
    if ("interview_date" %in% names(finalTable)) {
      # Try converting to Date if possible. If your data includes time-of-day,
      # you might use as.POSIXct instead.
      finalTable$interview_date <- as.Date(finalTable$interview_date, format="%m/%d/%Y")
    } else {
      # If no interview_date at all, you can handle differently
      warning("No interview_date column found after merging all files.")
    }
    
    # Identify placeholder dates from the MATLAB code:
    invalid_dates <- as.Date(c("1903-03-03", "1909-09-09", "1901-01-01", 
                               "1909-09-01", "1901-09-09"))
    
    # Convert any onset columns to Date; remove invalid placeholders -> NA
    # We'll define a small helper to do that:
    clean_onset_dates <- function(vec) {
      # Convert to Date if not already
      # If it’s already character, convert:
      if (is.character(vec)) {
        vec <- as.Date(vec)
      }
      # If it’s numeric or something else, may need special conversion
      # For demonstration, assume character or already Date.
      
      # Replace placeholders with NA
      vec[vec %in% invalid_dates] <- NA
      return(vec)
    }
    
    # Convert onset_vars to Date, removing placeholders
    for (colname in onset_vars) {
      if (colname %in% names(finalTable)) {
        finalTable[[colname]] <- clean_onset_dates(finalTable[[colname]])
      }
    }
    
    # We'll define a small function to get the "minimum date or NA if all are NA"
    safe_row_min <- function(dates_row) {
      # If all are NA, return NA
      if (all(is.na(dates_row))) {
        return(NA_real_)
      } else {
        return(min(dates_row, na.rm = TRUE))
      }
    }
    
    #### 1) Duration of CHR symptoms
    # Onset is the min date across onset_vars
    if (length(onset_vars) > 0) {
      # Construct a matrix (or data frame) of the onset columns
      onset_df <- finalTable[ , onset_vars, drop = FALSE]
      # Apply row-wise to get min date
      chr_symptoms_onset <- apply(onset_df, 1, safe_row_min)
      # Convert to Date
      chr_symptoms_onset <- as.Date(chr_symptoms_onset, format = "%m/%d/%Y")
      
      finalTable$chr_symptoms_onset <- chr_symptoms_onset
      
      # Now compute difference in years
      finalTable$chr_symptoms_duration <- NA_real_
      if ("interview_date" %in% names(finalTable)) {
        # Using difftime in days -> convert to years (~365.25)
        finalTable$chr_symptoms_duration <-
          as.numeric(difftime(finalTable$interview_date,
                              finalTable$chr_symptoms_onset,
                              units = "days")) / 365.25
      }
    }
    
    #### 2) Lifetime APS: use sips_aps_lifetime to mask the onset columns
    if (length(sips_aps_lifetime) > 0) {
      # Step A: read the boolean columns (these might be 0/1)
      lifetime_aps <- finalTable[ , sips_aps_lifetime, drop = FALSE]
      # Replace NA with 0
      lifetime_aps[is.na(lifetime_aps)] <- 0
      # Convert to logical
      lifetime_aps <- lifetime_aps == 1
      
      # copy the onset DF
      onset_df <- finalTable[ , onset_vars, drop = FALSE]
      
      # For each row/col, if lifetime_aps is FALSE, onset is set to NA
      # We'll do this by iteration or by indexing:
      for (r in seq_len(nrow(onset_df))) {
        false_cols <- which(!as.logical(lifetime_aps[r, ]))
        onset_df[r, false_cols] <- NA
      }
      
      # Take the min across each row
      lifetime_aps_onset <- apply(onset_df, 1, safe_row_min)
      lifetime_aps_onset <- as.Date(lifetime_aps_onset, format = "%m/%d/%Y")
      finalTable$lifetime_aps_onset <- lifetime_aps_onset
      
      finalTable$duration_of_lifetime_aps <- NA_real_
      if ("interview_date" %in% names(finalTable)) {
        finalTable$duration_of_lifetime_aps <-
          as.numeric(difftime(finalTable$interview_date,
                              finalTable$lifetime_aps_onset,
                              units = "days")) / 365.25
      }
    }
    
    #### 3) Lifetime APS + CAARMS (the MATLAB code used combined_mask = lifetime_aps & cpy)
    # cpy are the caarms_pastyear columns
    if (length(caarms_pastyear) > 0 && length(sips_aps_lifetime) > 0) {
      # We already have lifetime_aps from above
      # We'll just re-construct it for clarity:
      lifetime_aps_df <- finalTable[ , sips_aps_lifetime, drop = FALSE]
      lifetime_aps_df[is.na(lifetime_aps_df)] <- 0
      lifetime_aps_df <- lifetime_aps_df == 1
      
      cpy <- finalTable[ , caarms_pastyear, drop = FALSE]
      cpy[is.na(cpy)] <- 0
      cpy <- cpy == 1
      
      combined_mask <- lifetime_aps_df & cpy  # Could also do OR if needed: (|)
      
      # Now we apply that mask to the onset columns
      onset_df <- finalTable[ , onset_vars, drop = FALSE]
      for (r in seq_len(nrow(onset_df))) {
        false_cols <- which(!combined_mask[r, ])
        onset_df[r, false_cols] <- NA
      }
      
      caarms_chr_onset <- apply(onset_df, 1, safe_row_min)
      caarms_chr_onset <- as.Date(caarms_chr_onset, format = "%m/%d/%Y")
      finalTable$caarms_chr_onset <- caarms_chr_onset
      
      finalTable$duration_of_caarms_chr <- NA_real_
      if ("interview_date" %in% names(finalTable)) {
        finalTable$duration_of_caarms_chr <-
          as.numeric(difftime(finalTable$interview_date,
                              finalTable$caarms_chr_onset,
                              units = "days")) / 365.25
      }
    }
    
    #### 4) Lifetime APS + SIPS Progression
    if (length(sips_aps_progression) > 0 && length(sips_aps_lifetime) > 0) {
      # Similar approach
      lifetime_aps_df <- finalTable[ , sips_aps_lifetime, drop = FALSE]
      lifetime_aps_df[is.na(lifetime_aps_df)] <- 0
      lifetime_aps_df <- lifetime_aps_df == 1
      
      sap <- finalTable[ , sips_aps_progression, drop = FALSE]
      sap[is.na(sap)] <- 0
      sap <- sap == 1
      
      combined_mask <- lifetime_aps_df & sap
      
      onset_df <- finalTable[ , onset_vars, drop = FALSE]
      for (r in seq_len(nrow(onset_df))) {
        false_cols <- which(!combined_mask[r, ])
        onset_df[r, false_cols] <- NA
      }
      
      sips_progression_onset <- apply(onset_df, 1, safe_row_min)
      sips_progression_onset <- as.Date(sips_progression_onset, format = "%m/%d/%Y")
      finalTable$sips_progression_onset <- sips_progression_onset
      
      finalTable$duration_of_sips_progression <- NA_real_
      if ("interview_date" %in% names(finalTable)) {
        finalTable$duration_of_sips_progression <-
          as.numeric(difftime(finalTable$interview_date,
                              finalTable$sips_progression_onset,
                              units = "days")) / 365.25
      }
    }
    
    # Return the final data frame
    return(finalTable)
  }
  
  
  
  # helper that knows how to load each "file"
  read_source <- function(src) {
    # 1) PennCNB “file”
    if (grepl("penncnb01", src, ignore.case = TRUE)) {
      return(penn_collapsed)
    }
    
    # 2) if src already points to a file on disk, use it
    if (file.exists(src)) {
      return(read_csv_cached(src))
    }
    
    # 3) otherwise assume it’s a basename under data_DIR
    return(read_csv_cached(file.path(data_DIR, src)))
  }
  
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # 2) Preliminary checks, dictionary load, etc. -------
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  if (!dir.exists(data_DIR)) {
    stop("❌ Directory not found: ", data_DIR)
  }
  
  
  
  selected_variables <- variable_selection(dictionary_DIR, 
                                           required_vars = c("phenotype","group"))
  vars          <- selected_variables[[1]]  # The main variable list for basetable
  meta_table    <- selected_variables[[4]]  # meta info for data sources
  
  # Detect whether Prescient or AMP-SCZ structure
  all_csv_files <- list.files(data_DIR, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
  is_prescient_folder <- any(grepl("\\d{2}\\.\\d{2}\\.\\d{4}", all_csv_files))
  is_ampscz_folder    <- any(grepl("/csv/\\d{5,}/", all_csv_files))
  
  message("🔎 Detected Data: ", ifelse(is_prescient_folder, "Prescient", "AmpScz"))
  
  # A small caching list so each file is read only once
  .file_cache <- list()
  read_csv_cached <- function(fpath, ...) {
    if (!fpath %in% names(.file_cache)) {
      .file_cache[[fpath]] <<- withCallingHandlers(
        read_csv(fpath, show_col_types = FALSE, ...),
        warning = function(w) {
          invokeRestart("muffleWarning")
        }
      )
    }
    .file_cache[[fpath]]
  }
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # 3) PRESCIENT BRANCH ----
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  if (is_prescient_folder) {
    setwd(data_DIR)
    
    ## Extract date from folder name ----
    DATE <- str_extract(data_DIR, "\\d{2}\\.\\d{2}\\.\\d{4}")
    
    
    manage <- c(-9, -99, 999) #Which values have to be set to NA
    
    ## 3.a) Calculate scores ----
    # -- cdss --
    cdss_file <- file.path(data_DIR, paste0("PrescientStudy_Prescient_cdss_", DATE, ".csv"))
    cdss <- read_csv_cached(cdss_file)
    cdss_matrix <- cdss %>%
      select(starts_with("chrcdss_calg")) %>%
      as.data.frame()
    cdss_matrix <- manage_missing(cdss_matrix, manage)  # same helper usage
    cdss$cdss_total <- calc_total(cdss_matrix)
    updated_cdss_file <- file.path(data_DIR, paste0("PrescientStudy_Prescient_cdss_totalupdated_", DATE, ".csv"))
    write_csv(cdss, updated_cdss_file)
    
    # -- nsipr --
    nsipr_file <- file.path(data_DIR, paste0("PrescientStudy_Prescient_nsipr_", DATE, ".csv"))
    nsipr <- read_csv_cached(nsipr_file)
    nsipr_matrix <- nsipr %>%
      select(starts_with("chrnsipr_item")) %>%
      as.data.frame()
    nsipr_matrix <- manage_missing(nsipr_matrix, manage)
    nsipr$nsipr_total <- calc_total(nsipr_matrix)
    updated_nsipr_file <- file.path(data_DIR, paste0("PrescientStudy_Prescient_nsipr_totalupdated_", DATE, ".csv"))
    write_csv(nsipr, updated_nsipr_file)
    
    # -- oasis --
    oasis_file <- file.path(data_DIR, paste0("PrescientStudy_Prescient_oasis_", DATE, ".csv"))
    oasis <- read_csv_cached(oasis_file)
    oasis_matrix <- oasis %>%
      select(starts_with("chroasis_oasis")) %>%
      as.data.frame()
    oasis_matrix <- manage_missing(oasis_matrix, manage)
    oasis$oasis_total <- calc_total(oasis_matrix)
    updated_oasis_file <- file.path(data_DIR, paste0("PrescientStudy_Prescient_oasis_totalupdated_", DATE, ".csv"))
    write_csv(oasis, updated_oasis_file)
    
    # -- chr duration --
    fileNames <- c(
      file.path(data_DIR, paste0("PrescientStudy_Prescient_psychs_p1p8_", DATE, ".csv")),
      file.path(data_DIR, paste0("PrescientStudy_Prescient_psychs_p9ac32_", DATE, ".csv"))
    )
    duration_of_chr <- calculate_chr_duration(fileNames)
    chr_dur_file <- file.path(data_DIR, paste0("PrescientStudy_Prescient_psychs_CHR-duration_", DATE, ".csv"))
    write_csv(duration_of_chr, chr_dur_file)
    
    # -- get baseline PGIS only --
    pgis_file <- file.path(data_DIR, paste0("PrescientStudy_Prescient_pgis_", DATE, ".csv"))
    pgis <- read_csv_cached(pgis_file)
    pgis_baseline <- pgis %>% filter(visit == 2)
    updated_pgis_file <- file.path(data_DIR, paste0("PrescientStudy_Prescient_pgis_baseline_", DATE, ".csv"))
    write_csv(pgis_baseline, updated_pgis_file)
    
    ## 3.b) Now define the variables to extract ----
    csv_files <- list.files(data_DIR, pattern = "*.csv", full.names = TRUE)
    file_info <- data.frame(
      file_path = csv_files,
      file_name = basename(csv_files),
      stringsAsFactors = FALSE
    )
    extract_date <- function(filename) {
      date_match <- str_extract(filename, "\\d{2}\\.\\d{2}\\.\\d{4}")
      if (!is.na(date_match)) as.Date(date_match, format = "%m.%d.%Y") else NA
    }
    file_info$date <- sapply(file_info$file_name, extract_date)
    file_info$base_name <- str_replace(file_info$file_name, "_\\d{2}\\.\\d{2}\\.\\d{4}\\.csv", "")
    latest_files <- file_info %>%
      group_by(base_name) %>%
      filter(date == max(date, na.rm = TRUE)) %>%
      ungroup() %>%
      pull(file_path)
    
    # Initialize list for storing matches
    vars_list <- list()
    for (file in latest_files) {
      file_vars <- tryCatch({
        colnames(read.csv(file, nrows = 1, check.names = FALSE, fill = TRUE, skipNul = TRUE))
      }, error = function(e) NULL)
      if (is.null(file_vars)) next
      matched_vars <- intersect(file_vars, vars)
      if (length(matched_vars) > 0) {
        vars_list[[file]] <- matched_vars
      } else {
        message("ℹ️ No matches found in: ", file, " - Skipping")
      }
    }
    
    # Convert list to a data frame
    if (length(vars_list) > 0) {
      vars <- do.call(rbind, lapply(names(vars_list), function(file) {
        data.frame(
          file = gsub(".*/", "", file),
          var  = vars_list[[file]],
          stringsAsFactors = FALSE
        )
      }))
    }
    
    ## 3.c) Collect all subjectkeys from all relevant CSV files ----
    all_subjects <- c()
    for(i in seq_len(nrow(vars))) {
      csv_name <- vars$file[i]
      fpath <- file.path(data_DIR, csv_name)
      temp_df <- read_csv_cached(fpath)
      all_subjects <- c(all_subjects, temp_df$subjectkey)
    }
    unique_subjects <- unique(all_subjects)
    unique_subjects <- unique_subjects[!(unique_subjects == "" | is.na(unique_subjects))]
    
    ############
    ## 3.d) Build final table T----
    T <- data.frame(subjectkey = unique_subjects, stringsAsFactors = FALSE)
    
    ## 3.e) Process each unique variable efficiently----
    unique_vars <- unique(vars$var)
    
    for (var_name in unique_vars) {
      ## Get all files that contain this variable
      files_with_var <- vars$file[vars$var == var_name]
      
      combined_dfs <- lapply(files_with_var, function(csv) {
        df <- read_csv_cached(csv) %>% 
          filter(!is.na(subjectkey) & subjectkey != "")
        
        
        ## Safely convert interview_date if it exists
        if ("interview_date" %in% names(df)) {
          df$interview_date <- ifelse(
            grepl("^[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}$", df$interview_date),
            as.character(df$interview_date),  # keep the string
            NA
          )
          df <- df %>% 
            mutate(interview_date = suppressWarnings(as.Date(interview_date, tryFormats = c("%d/%m/%Y"))))
        }
        
      })
      
      ## Combine all dataframes into one, ensuring type consistency
      combined_df <- bind_rows(combined_dfs)
      
      ## Keep only the earliest interview_date for each subject if it exists
      if ("interview_date" %in% names(combined_df)) {
        combined_df <- combined_df %>% 
          filter(!is.na(interview_date)) %>% 
          group_by(subjectkey) %>% 
          slice_min(order_by = interview_date, n = 1, with_ties = FALSE) %>% 
          ungroup()
      }
      
      ## If variable does not exist in T, initialize with NA
      if (!(var_name %in% names(T))) {
        T[[var_name]] <- NA
      }
      
      ## Merge data into T efficiently
      if (var_name %in% names(combined_df)) {
        matched_values <- combined_df[[var_name]][match(T$subjectkey, combined_df$subjectkey)]
        T[[var_name]] <- ifelse(is.na(matched_values), T[[var_name]], matched_values)
      }
    }
    
    
    
    #########
    
    ## 3.f) Convert placeholders to NA (vectorized)----
    manage_values_numeric <- c(-9, -99, 999, -900)
    T[] <- lapply(T, function(x) ifelse(x %in% manage_values_numeric, NA, x))
    
    manage_values_character <- c('-9', '-99', '999', '-900')
    T[] <- lapply(T, function(x) ifelse(x %in% manage_values_character, NA, x))
    
    
    
    ## 3.g) Exclude test cases ----
    clientlist_file <- file.path(data_DIR, paste0("PrescientStudy_Prescient_ClientListWithDpaccID_", DATE, ".csv"))
    if (file.exists(clientlist_file)) {
      T_site <- read_csv_cached(clientlist_file)
      testcases <- T_site$subjectkey[T_site$fkLocationID == 13]
      T <- T[!(T$subjectkey %in% testcases), ]
    }
    T <- T[!(grepl("^ZZ", T$subjectkey) | grepl("^TE", T$subjectkey)), ]
    
    
    ## 3.h) Additional recoding ----
    if ("interview_age" %in% names(T) && is.numeric(T$interview_age)) {
      T$interview_age <- T$interview_age / 12
    }
    if ("chrscid_a25" %in% names(T) && is.numeric(T$chrscid_a25)) {
      T$chrscid_a25[T$chrscid_a25 < 3] <- 0
      T$chrscid_a25[T$chrscid_a25 == 3] <- 1
    }
    old_name <- "chrdemo_racial_back____9"
    new_name <- "chrdemo_racial_back___9"
    if(old_name %in% names(T)) {
      names(T)[names(T) == old_name] <- new_name
    }
    
    T$chrdemo_racial_back <- rep(NA, nrow(T))
    for(i in 1:9) {
      col_name <- paste0("chrdemo_racial_back___", i)
      if(col_name %in% names(T)) {
        T$chrdemo_racial_back[T[[col_name]] == 1] <- i
      }
    }
    if ("chrdemo_working" %in% names(T) && is.numeric(T$chrdemo_working)) {
      T$chrdemo_working[T$chrdemo_working == 2] <- 1
    }
    
    ## 3.i) Make separate column for sites----
    subject_id_mapping <- data.frame(
      Abbreviation = c("BI","BM","CA","CG","CM","CP","GA","GW","HA","HK",
                       "IR","JE","KC","LA","LS","MA","ME","MT","MU","NC",
                       "NL","NN","OH","OR","PA","PI","PV","SD","SF","SG",
                       "SH","SI","SL","ST","TE","UR","WU","YA"),
      FullName = c("Beth_Israel_(Harvard)","Birmingham","Calgary_(Canada)","Cologne",
                   "Cambridge_(UK)","Copenhagen","Georgia","Gwangju",
                   "Hartford_(Institute_of_Living)","Hong_Kong",
                   "UC_Irvine","Jena","King's_College_(UK)","UCLA","Lausanne",
                   "Madrid_(Spain)","Melbourne","Montreal_(Canada)","Munich_(Germany)","UNC",
                   "Northwell","Northwestern","Ohio","Oregon","Uni_of_Pennsylvania",
                   "Pittsburgh_(UPMC)","Pavia_(Italy)","UCSD","UCSF","Singapore",
                   "Shanghai_(China)","Mt._Sinai","Seoul_(South_Korea)","Santiago",
                   "Temple","Uni_of_Rochester","Washington_University","Yale"),
      stringsAsFactors = FALSE
    )
    T <- T %>%
      mutate(Tempstore = substr(subjectkey, 1, 2)) %>%
      left_join(subject_id_mapping, by=c("Tempstore"="Abbreviation")) %>%
      rename(Site = FullName) %>%
      select(-Tempstore)
    
    ## 3.j) Exclude cases with no data except subjectkey & group (vectorized)----
    non_key_cols <- setdiff(names(T), c("subjectkey", "group"))
    T <- T[rowSums(!is.na(T[, non_key_cols])) > 0, ]
    
    ## 3.k) Identify ineligible cases ----
    incex_file <- file.path(data_DIR, paste0("PrescientStudy_Prescient_inclusionexclusion_criteria_review_", DATE, ".csv"))
    T$ineligible_index <- FALSE
    if (file.exists(incex_file)) {
      T_incex <- read_csv_cached(incex_file)
      if ("chrcrit_excluded" %in% names(T_incex)) {
        cases_ineligible <- T_incex$subjectkey[as.logical(T_incex$chrcrit_excluded)]
        ineligible_indices <- match(cases_ineligible, T$subjectkey)
        ineligible_indices <- ineligible_indices[!is.na(ineligible_indices)]
        T$ineligible_index[ineligible_indices] <- TRUE
      }
    }
    
    ## 3.l) Fix "ME67990" group, create binary_group----
    idx_ME67990 <- which(T$subjectkey == "ME67990")
    if(length(idx_ME67990) == 1) {
      T$group[idx_ME67990] <- "UHR"
    }
    T$binary_group <- ifelse(T$group == "UHR", 1, 0)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## 3.m) Remove columns that have no data (all NA or all "")----
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    empty_cols <- sapply(T, function(col) {
      if (is.character(col)) {
        all(is.na(col) | col == "")
      } else if (is.factor(col)) {
        # Convert factor to character for the comparison
        all(is.na(col) | as.character(col) == "")
      } else {
        all(is.na(col))
      }
    })
    if (any(empty_cols)) {
      removed_vars <- names(T)[empty_cols]
      T <- T[, !empty_cols, drop = FALSE]
      warning(
        "The following variables had no data at all and have been removed from the basetable: ",
        paste(removed_vars, collapse = ", ")
      )
    }
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## 3.n) Change datatype based on notes in dictionary----
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the dictionary
    dict <- readxl::read_excel(dictionary_DIR, col_types = "text")
    ignore_keys <- c(-900, -300)
    
    # Infer datatype inline: if a note contains at least two mapping pairs with keys other than -900 or -300, mark it as Categorical; otherwise, continuous.
    dict$InferredDataType <- sapply(dict$Notes, function(note) {
      pairs <- str_extract_all(note, "(-?\\d+)\\s*=\\s*[^,;]+")[[1]]
      if (length(pairs) == 0) return(NA)
      keys <- as.numeric(str_extract(pairs, "-?\\d+"))
      valid_keys <- unique(keys[!(keys %in% ignore_keys)])
      if (length(valid_keys) >= 2) "Categorical" else NA
    })
    
    # Identify variables marked as "Categorical"x
    cat_vars <- dict$ElementName[dict$InferredDataType == "Categorical"]
    
    # Convert the corresponding columns in T to character
    cols_to_convert <- names(T) %in% cat_vars
    T[cols_to_convert] <- lapply(T[cols_to_convert], as.character)
    
    
    #### Warning about non applicable data
    
    # Check each column for the value -3 or -300 and capture the column names that contain -3 or -300
    cols_with_neg3 <- names(T)[sapply(T, function(x) any(x %in% c(-3,-300,-7), na.rm=TRUE))]
    
    # Return TRUE if any column has -3, otherwise FALSE
    any_neg3 <- length(cols_with_neg3) > 0
    
    if (any_neg3){
      options(warning.length = 8170) # How many variables it will show
      warning(
        paste(
          "-3 and/or -300 and/or -7 was found in the data. -3, -7 and -300 are entered when the question was non applicable.",
          "For example, a question was skipped because of a previous question.",
          "Please look into your data and consider how to deal with these -3, -7 and/or -300 values.",
          "These variables contain -3/-7/-300:",
          paste(cols_with_neg3, collapse = ", "),
          sep = "\n"
        )
      )
    }
    
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## 3.o) Write out final table----
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    outdir <- file.path(data_DIR, "BASELINE_TABLES")
    dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
    outfile <- file.path(outdir, paste0("basetable_", Sys.Date(), ".csv"))
    write_csv(T, outfile)
    
    message("Basetable saved to: ", outfile)
    
    # Return final results
    return(list(data=T, meta=meta_table))
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4) AMP-SCZ release 2 BRANCH----
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
  } else if (is_ampscz_folder && release=="Amp_scz2") {
    
    ## 4.a) Calculate totals for nsipr----
    subfolders <- list.dirs(file.path(data_DIR, "ampscz_nsipr01/csv"), recursive = FALSE)
    version_numbers <- as.numeric(str_extract(subfolders, "\\d+$"))
    latest_version <- max(version_numbers, na.rm = TRUE)
    latest_folder <- file.path(data_DIR, "ampscz_nsipr01/csv", latest_version)
    nsipr_file <- list.files(latest_folder, pattern = "part-.*\\.csv$", full.names = TRUE)
    
    nsipr <- read_csv_cached(nsipr_file[1])
    nsipr_matrix <- nsipr %>%
      select(starts_with("chrnsipr_item")) %>%
      as.data.frame()
    manage_values <- c(-9, -99, 999)
    nsipr_matrix[nsipr_matrix %in% manage_values] <- NA
    nsipr$nsipr_total <- rowSums(nsipr_matrix, na.rm = TRUE)
    output_file <- file.path(data_DIR, "PrescientStudy_Prescient_nsipr_totalupdated.csv")
    write_csv(nsipr, output_file)
    
    ## 4.b) find variables in files----
    all_csv_files <- list.files(data_DIR, pattern = "*.csv", full.names = TRUE, recursive = TRUE)
    all_csv_files <- grep("BASELINE_TABLES", all_csv_files, invert = TRUE, value = TRUE)    
    file_info <- data.frame(
      file_path = all_csv_files,
      dataset_folder = dirname(all_csv_files),
      stringsAsFactors = FALSE
    )
    file_info$version <- as.numeric(str_extract(file_info$dataset_folder, "\\d+$"))
    file_info$version[is.na(file_info$version)] <- 9999
    file_info$datatype_folder <- str_extract(file_info$dataset_folder, paste0("^", data_DIR, "/[^/]+"))
    file_info$datatype <- basename(file_info$datatype_folder)
    
    latest_files <- file_info %>%
      group_by(datatype) %>%
      filter(version == max(version, na.rm = TRUE)) %>%
      ungroup() %>%
      pull(file_path)
    
    vars_list <- list()
    for (file in latest_files) {
      file_vars <- tryCatch({
        colnames(read.csv(file, nrows = 1, check.names = FALSE, fill = TRUE, skipNul = TRUE))
      }, error = function(e) NULL)
      if (is.null(file_vars)) next
      matched_vars <- intersect(file_vars, vars)
      if (length(matched_vars) > 0) {
        vars_list[[file]] <- matched_vars
      } else {
        message("ℹ️ No matches found in: ", file, " - Skipping")
      }
    }
    
    if (length(vars_list) > 0) {
      vars <- do.call(rbind, lapply(names(vars_list), function(file) {
        data.frame(
          file = file,
          var  = vars_list[[file]],
          stringsAsFactors = FALSE
        )
      }))
    }
    
    ## 4.c) Collect all subject IDs----
    all_subjects <- c()
    for(i in seq_len(nrow(vars))) {
      csv <- vars$file[i]
      temp_df <- read_csv_cached(csv)
      all_subjects <- c(all_subjects, temp_df$src_subject_id)
    }
    unique_subjects <- unique(all_subjects)
    unique_subjects <- unique_subjects[!(unique_subjects == "" | is.na(unique_subjects))]
    
    
    
    ## 4.d) Build final table T----
    T <- data.frame(src_subject_id = unique_subjects, stringsAsFactors = FALSE)
    
    ## 4.e) Process each unique variable efficiently----
    unique_vars <- unique(vars$var)
    
    for (var_name in unique_vars) {
      ## Get all files that contain this variable
      files_with_var <- vars$file[vars$var == var_name]
      
      combined_dfs <- lapply(files_with_var, function(csv) {
        df <- read_csv_cached(csv) %>% 
          filter(!is.na(src_subject_id) & src_subject_id != "")
        
        
        ## Safely convert interview_date if it exists
        if ("interview_date" %in% names(df)) {
          df <- df %>% 
            mutate(interview_date = suppressWarnings(as.Date(interview_date, tryFormats = c("%d/%m/%Y"))))
        }
        
      })
      
      ## Combine all dataframes into one, ensuring type consistency
      combined_df <- bind_rows(combined_dfs)
      
      ## Keep only the earliest interview_date for each subject if it exists
      if ("interview_date" %in% names(combined_df)) {
        combined_df <- combined_df %>% 
          filter(!is.na(interview_date)) %>% 
          group_by(src_subject_id) %>% 
          slice_min(order_by = interview_date, n = 1, with_ties = FALSE) %>% 
          ungroup()
      }
      
      ## If variable does not exist in T, initialize with NA
      if (!(var_name %in% names(T))) {
        T[[var_name]] <- NA
      }
      
      ## Merge data into T efficiently
      if (var_name %in% names(combined_df)) {
        matched_values <- combined_df[[var_name]][match(T$src_subject_id, combined_df$src_subject_id)]
        T[[var_name]] <- ifelse(is.na(matched_values), T[[var_name]], matched_values)
      }
    }
    
    
    ## 4.f) Convert placeholders to NA (vectorized)----
    manage_values_numeric <- c(-9, -99, 999, -900)
    T[] <- lapply(T, function(x) ifelse(x %in% manage_values_numeric, NA, x))
    
    manage_values_character <- c('-9', '-99', '999', '-900')
    T[] <- lapply(T, function(x) ifelse(x %in% manage_values_character, NA, x))
    
    
    ## 4.g) Additional recodes----
    if("interview_age" %in% names(T) && is.numeric(T$interview_age)) {
      T$interview_age <- T$interview_age / 12
    }
    old_name <- "chrdemo_racial_back____9"
    new_name <- "chrdemo_racial_back___9"
    if(old_name %in% names(T)) {
      names(T)[names(T) == old_name] <- new_name
    }
    
    if(any(grepl("^chrdemo_racial_back___\\d", names(T)))) {
      T$chrdemo_racial_back <- NA
      for (i in 1:9) {
        col_name <- paste0("chrdemo_racial_back___", i)
        if (col_name %in% names(T)) {
          T$chrdemo_racial_back[T[[col_name]] == 1] <- i
        }
      }
    }
    if("chrdemo_working" %in% names(T) && is.numeric(T$chrdemo_working)) {
      T$chrdemo_working[T$chrdemo_working == 2] <- 1
    }
    
    
    
    ## 4.h) Exclude cases with no data except subjectkey & group----
    non_key_cols <- setdiff(names(T), c("src_subject_id", "group"))
    T <- T[rowSums(!is.na(T[, non_key_cols])) > 0, ]
    
    
    ## 4.i) Site column----
    subject_id_mapping <- data.frame(
      Abbreviation = c("BI","BM","CA","CG","CM","CP","GA","GW","HA","HK",
                       "IR","JE","KC","LA","LS","MA","ME","MT","MU","NC",
                       "NL","NN","OH","OR","PA","PI","PV","SD","SF","SG",
                       "SH","SI","SL","ST","TE","UR","WU","YA"),
      FullName = c("Beth_Israel_(Harvard)","Birmingham","Calgary_(Canada)","Cologne",
                   "Cambridge_(UK)","Copenhagen","Georgia","Gwangju",
                   "Hartford_(Institute_of_Living)","Hong_Kong",
                   "UC_Irvine","Jena","King's_College_(UK)","UCLA","Lausanne",
                   "Madrid_(Spain)","Melbourne","Montreal_(Canada)","Munich_(Germany)","UNC",
                   "Northwell","Northwestern","Ohio","Oregon","Uni_of_Pennsylvania",
                   "Pittsburgh_(UPMC)","Pavia_(Italy)","UCSD","UCSF","Singapore",
                   "Shanghai_(China)","Mt._Sinai","Seoul_(South_Korea)","Santiago",
                   "Temple","Uni_of_Rochester","Washington_University","Yale"),
      stringsAsFactors = FALSE
    )
    T <- T %>%
      mutate(Tempstore = substr(src_subject_id, 1, 2)) %>%
      left_join(subject_id_mapping, by = c("Tempstore" = "Abbreviation")) %>%
      rename(Site = FullName) %>%
      select(-Tempstore)
    
    ## 4.j) Create binary group----
    T$group <- NA
    T$group[T$phenotype == "CHR"] <- 1
    T$group[T$phenotype == "HC"]  <- 0
    
    ## remove rows with no group
    T <- T[!is.na(T$group), ]
    
    
    
    ## 4.k) Recode data----
    # Loop over every column in data and apply recode_if_present
    for (col in names(T)) {
      if (length(na.omit(T[[col]]))>0){
        T <- recode_if_present(T, col)
      }
    }    
    
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## 4.l) Remove columns that have no data (all NA or all "")----
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    empty_cols <- sapply(T, function(col) {
      if (is.character(col)) {
        all(is.na(col) | col == "")
      } else if (is.factor(col)) {
        # Convert factor to character for the comparison
        all(is.na(col) | as.character(col) == "")
      } else {
        all(is.na(col))
      }
    })
    if (any(empty_cols)) {
      removed_vars <- names(T)[empty_cols]
      T <- T[, !empty_cols, drop = FALSE]
      warning(
        "The following variables had no data at all and have been removed from the basetable: ",
        paste(removed_vars, collapse = ", ")
      )
    }
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## 4.m) Change datatype based on notes in dictionary----
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the dictionary
    dict <- readxl::read_excel(dictionary_DIR, 
                               col_types = "text")
    ignore_keys <- c(-900, -300)
    
    # Infer datatype inline: if a note contains at least two mapping pairs with keys other than -900 or -300, mark it as Categorical; otherwise, continuous.
    dict$InferredDataType <- sapply(dict$Notes, function(note) {
      pairs <- str_extract_all(note, "(-?\\d+)\\s*=\\s*[^,;]+")[[1]]
      if (length(pairs) == 0) return(NA)
      keys <- as.numeric(str_extract(pairs, "-?\\d+"))
      valid_keys <- unique(keys[!(keys %in% ignore_keys)])
      if (length(valid_keys) >= 2) "Categorical" else NA
    })
    
    # Identify variables marked as "Categorical"
    cat_vars <- dict$ElementName[dict$InferredDataType == "Categorical"]
    
    # Convert the corresponding columns in T to character
    cols_to_convert <- names(T) %in% cat_vars
    T[cols_to_convert] <- lapply(T[cols_to_convert], as.character)
    
    
    #### Warning about non applicable data
    
    # Check each column for the value -3 or -300 and capture the column names that contain -3 or -300
    cols_with_neg3 <- names(T)[sapply(T, function(x) any(x %in% c(-3, -7,-300), na.rm=TRUE))]
    
    # Return TRUE if any column has -3, otherwise FALSE
    any_neg3 <- length(cols_with_neg3) > 0
    
    if (any_neg3){
      options(warning.length = 8170) # How many variables it will show
      warning(
        paste(
          "-3, =7 and/or -300 was found in the data. -3, -7 and -300 are entered when the question was non applicable.",
          "For example, a question was skipped because of a previous question.",
          "Please look into your data and consider how to deal with these -3, -7 and/or -300 values.",
          "These variables contain -3/-7/-300:",
          paste(cols_with_neg3, collapse = ", "),
          sep = "\n"
        )
      )
    }
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## 4.n) Write out final table----
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    outdir <- file.path(data_DIR, "BASELINE_TABLES")
    dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
    outfile <- file.path(outdir, paste0("Rbasetable_", Sys.Date(), ".csv"))
    outfile_meta <- file.path(outdir, paste0("metatable_", Sys.Date(), ".csv"))
    write_csv(T, outfile)
    write_csv(meta_table, outfile_meta)
    
    message("Basetable saved to: ", outfile)
    
    return(list(data=T, meta=meta_table))
    
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5) AMP-SCZ release 3 BRANCH----
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
  }else if (release=="Amp_scz3") {
    
    ## 5.a) Calculate totals for nsipr----

    nsipr_file <- file.path(data_DIR, "ampscz_nsipr01/csv/part-00000-e0ef9a8c-f645-43e4-a950-fc6627d573e4-c000.csv")
    
    nsipr <- read_csv_cached(nsipr_file)
    nsipr_matrix <- nsipr %>%
      select(starts_with("chrnsipr_item")) %>%
      as.data.frame()
    manage_values <- c(-9, -99, 999)
    nsipr_matrix[nsipr_matrix %in% manage_values] <- NA
    nsipr$nsipr_total <- rowSums(nsipr_matrix, na.rm = TRUE)
    output_file <- file.path(data_DIR, "PrescientStudy_Prescient_nsipr_totalupdated.csv")
    write_csv(nsipr, output_file)
    
    ## 5.b) find variables in files----
    all_csv_files <- list.files(data_DIR, pattern = "*.csv", full.names = TRUE, recursive = TRUE)
    all_csv_files <- grep("BASELINE_TABLES", all_csv_files, invert = TRUE, value = TRUE)
    all_csv_files <- grep("MONTH1_TABLES", all_csv_files, invert = TRUE, value = TRUE)    
    all_csv_files <- grep("MONTH2_TABLES", all_csv_files, invert = TRUE, value = TRUE)  
    all_csv_files <- grep("SCREENING_TABLES", all_csv_files, invert = TRUE, value = TRUE)    
    file_info <- data.frame(
      file_path = all_csv_files,
      dataset_folder = dirname(all_csv_files),
      stringsAsFactors = FALSE
    )
    file_info$version <- as.numeric(str_extract(file_info$dataset_folder, "\\d+$"))
    file_info$version[is.na(file_info$version)] <- 9999
    file_info$datatype_folder <- str_extract(file_info$dataset_folder, paste0("^", data_DIR, "/[^/]+"))
    file_info$datatype <- basename(file_info$datatype_folder)
    
    latest_files <- file_info$file_path
    
    vars_list <- list()
    for (file in latest_files) {
      file_vars <- tryCatch({
        colnames(read.csv(file, nrows = 1, check.names = FALSE, fill = TRUE, skipNul = TRUE))
      }, error = function(e) NULL)
      if (is.null(file_vars)) next
      matched_vars <- intersect(file_vars, vars)
      if (length(matched_vars) > 0) {
        vars_list[[file]] <- matched_vars
      } else {
        message("ℹ️ No matches found in: ", file, " - Skipping")
      }
    }
    
    #### --- 5.b.1) special PennCNB handling ------------------------------------
    penn_folder <- file.path(data_DIR, "penncnb01", "csv")
    if (dir.exists(penn_folder)) {
      # 1) read & bind every PennCNB CSV
      penn_files <- list.files(penn_folder, pattern = "\\.csv$", full.names = TRUE)
      penn_raw   <- lapply(penn_files, read_csv_cached) %>% bind_rows()
      
      # 2) restrict to your visit of interest, if present
      if ("visit" %in% names(penn_raw)) {
        penn_raw <- penn_raw %>% filter(visit == timepoint)
      }
      
      # 3) collapse to one row per subject
      penn_collapsed <- penn_raw %>%
        group_by(subjectkey) %>%
        summarise(
          across(
            everything(),
            ~ if (all(is.na(.x))) {
              NA
            } else {
              first(na.omit(.x))
            }
          ),
          .groups = "drop"
        )
      
      # 4) pick out which of your `vars` actually live in PennCNB
      penn_vars <- intersect(vars, colnames(penn_collapsed))
      if (length(penn_vars) > 0) {
        # we’ll pretend they all came from a single “file”
        vars_list[["penncnb01_collapsed"]] <- penn_vars
      }
    }

    # now convert vars_list -> your vars data.frame exactly as before
    if (length(vars_list) > 0) {
      vars <- do.call(rbind, lapply(names(vars_list), function(file) {
        data.frame(
          file = file,
          var  = vars_list[[file]],
          stringsAsFactors = FALSE
        )
      }))
    }
    
    ## 5.c) Collect all subject IDs, fast and minimal columns ----
    
    # only read each file once
    files_to_read <- unique(vars$file)
    
    id_lists <- lapply(files_to_read, function(src) {
      if (grepl("penncnb01", src, ignore.case = TRUE)) {
        # penn_collapsed is already in memory, just grab the column
        read_source(src)$subjectkey   # or $src_subject_id in the AMP-SCZ branch
      } else {
        # for normal CSVs, read only the ID column
        read_csv(
          file.path(src),
          col_types = cols_only(src_subject_id = col_character())
        )$src_subject_id
      }
    })
    
    # flatten and dedupe, drop blanks/NA
    unique_subjects <- unique(unlist(id_lists, use.names = FALSE))
    unique_subjects <- unique_subjects[unique_subjects != "" & !is.na(unique_subjects)]
    
    
    ## 5.d) Build final table T----
    T <- data.frame(src_subject_id = unique_subjects, stringsAsFactors = FALSE)
    
    
    ## 5.e) Process each unique variable in TWO PASSES, to ensure dates closest to timepoint are set first ----
    
    all_vars <- unique(vars$var)
    
    #-----------------#
    #  PASS 1:        #
    #  Variables that #
    #  DO have visit column  
    #-----------------#
    visit_vars   <- c() # We'll track which var_names actually had a 'visit' column
    no_visit_vars <- c()
    
    # We'll store (subject => earliest interview_date) from any variable's file that has visit==timepoint
    # The idea: each subject might have multiple potential Match dates across different variables/files;
    # we keep the earliest. Then we have a single "Match" reference per subject.
    Match_dates <- data.frame(
      src_subject_id = character(),
      interview_date = as.Date(character(), format = "%m/%d/%Y"),
      stringsAsFactors = FALSE
    )
    
    
    for (var_name in all_vars) {
      files_with_var <- vars$file[vars$var == var_name]
      
      #–– special case: if one of our "files" is the penn_collapsed blob, use its names() ––
      if (any(grepl("penncnb01", files_with_var))) {
        any_visit_column <- "visit" %in% names(penn_collapsed)
      } else {
        any_visit_column <- FALSE
        for (file in files_with_var) {
          df_header <- tryCatch(
          read.csv(file, nrows = 1, check.names = FALSE),
          error = function(e) NULL
        )
        if (!is.null(df_header) && "visit" %in% names(df_header)) {
          any_visit_column <- TRUE
          break
        }
      }
        
      }      
      if (any_visit_column) {
        # We'll process these files now (PASS 1).
        # We also store earliest interview_date for Match.
        
        visit_vars <- c(visit_vars, var_name)  # track it
        
        # Merge (or overwrite) into T
        if (!(var_name %in% names(T))) {
          T[[var_name]] <- NA
        }
        
        for (file in files_with_var) {
          # now use the unified reader which knows about penn_collapsed
          df <- read_source(file) %>% 
            filter(!is.na(src_subject_id) & src_subject_id != "")
          
          if ("visit" %in% names(df)) {
            
            #For month 2 it's sometimes called differently so making sure that it's also considering the other option.
            if (timepoint == "month_2"){
              df <- df %>% filter(visit == timepoint | visit == "m2")
            } else {
              df <- df %>% filter(visit == timepoint)
              
            }

            
            # If the file has an interview_date, let’s store the earliest date per subject
            if ("interview_date" %in% names(df)) {
              df$interview_date <- as.Date(df$interview_date, format = "%m/%d/%Y")
              
              # Merge (subject => earliest interview_date) for any new subjects or if it's earlier
              for (i in seq_len(nrow(df))) {
                sid  <- df$src_subject_id[i]
                date <- df$interview_date[i]
                if (!is.na(date)) {
                  if (!sid %in% Match_dates$src_subject_id) {
                    # add new row
                    Match_dates <- rbind(Match_dates, 
                                            data.frame(src_subject_id=sid, 
                                                       interview_date=date,
                                                       stringsAsFactors=FALSE))
                  } else {
                    # check if it's earlier
                    olddate <- Match_dates$interview_date[Match_dates$src_subject_id == sid]
                    if (is.na(olddate) || date < olddate) {
                      Match_dates$interview_date[ Match_dates$src_subject_id == sid ] <- date
                    }
                  }
                }
              }
            }
            
            # Merge var_name data
            if (var_name %in% names(df)) {
              matched_vals <- df[[var_name]][ match(T$src_subject_id, df$src_subject_id) ]
              T[[var_name]] <- ifelse(is.na(matched_vals), T[[var_name]], matched_vals)
            }
          }
        }
        
      } else {
        # We'll handle these later, in PASS 2
        no_visit_vars <- c(no_visit_vars, var_name)
      }
    }
    
    #-----------------#
    #  PASS 2:        #
    #  Variables that #
    #  do NOT have    #
    #  a visit column #
    #-----------------#
    
    # For quick lookups: we store Match_dates as a named vector or keep it as DF
    # We'll keep it as a data frame but turn it into a quick index for merging.
    names(Match_dates)[2] <- "Match_date"  # so it's clear
    
    for (var_name in no_visit_vars) {
      files_with_var <- vars$file[vars$var == var_name]
      
      warning(paste(
        "No 'visit' variable found in any file for variable", var_name,
        "\nData will be used. If multiple interview_dates are available for this variable, the one closest to", timepoint,
        "is chosen. Please check the data of this variable if this is the correct approach. If not, manual adjustments are needed.",
        "\n"
      ))
      
      # We'll do "closest date" matching to each subject's Match date
      if (!(var_name %in% names(T))) {
        T[[var_name]] <- NA
      }
      
      for (file in files_with_var) {
        df <- read_csv_cached(file) %>%
          filter(!is.na(src_subject_id) & src_subject_id != "")
        
        # If the file has no interview_date column, we can't do "closest date" logic,
        # we just use the first row for each subject. (You could skip entirely if you prefer.)
        if (!("interview_date" %in% names(df))) {
          # So just do a direct match. If multiple rows for the same subject, last row wins
          # unless you do a group_by slice or something. We'll keep it simple:
          # We can do group_by src_subject_id => slice(1) to keep the first row, for instance:
          df <- df %>%
            group_by(src_subject_id) %>%
            slice(1) %>%
            ungroup()
          
          if (var_name %in% names(df)) {
            matched_vals <- df[[var_name]][ match(T$src_subject_id, df$src_subject_id) ]
            T[[var_name]] <- ifelse(is.na(matched_vals), T[[var_name]], matched_vals)
          }
          next
        }
        
        # Otherwise, do the "closest date" approach:
        df$interview_date <- as.Date(df$interview_date, format="%m/%d/%Y")
        
        # Join with Match_dates
        df <- df %>%
          left_join(Match_dates, by="src_subject_id") %>%
          mutate(
            diff_days = ifelse(
              !is.na(interview_date) & !is.na(Match_date),
              abs(as.numeric(difftime(interview_date, Match_date, units="days"))),
              NA
            )
          )
        
        # If Match_date is NA for a subject, that means we never had a 'visit' file for them
        # => diff_days is NA => tie for min => pick first row encountered
        df <- df %>%
          group_by(src_subject_id) %>%
          slice_min(order_by = diff_days, n = 1, with_ties = FALSE) %>%
          ungroup()
        
        if (var_name %in% names(df)) {
          matched_vals <- df[[var_name]][ match(T$src_subject_id, df$src_subject_id) ]
          T[[var_name]] <- ifelse(is.na(matched_vals), T[[var_name]], matched_vals)
        }
      }
    }
    
    
    
    ## 5.f) Convert placeholders to NA (vectorized)----
    manage_values_numeric <- c(-9, -99, 999)
    T[] <- lapply(T, function(x) ifelse(x %in% manage_values_numeric, NA, x))
    
    manage_values_character <- c('-9', '-99', '999', '-900')
    T[] <- lapply(T, function(x) ifelse(x %in% manage_values_character, NA, x))
    
    
    ## 5.g) Additional recodes----
    if("interview_age" %in% names(T) && is.numeric(T$interview_age)) {
      T$interview_age <- T$interview_age / 12
    }
  
    
    
    old_name <- "chrdemo_racial_back____9"
    new_name <- "chrdemo_racial_back___9"
    if(old_name %in% names(T)) {
      names(T)[names(T) == old_name] <- new_name
    }
    
    # Warn if user asked for the aggregate but forgot the item-level vars
    race_items <- paste0("chrdemo_racial_back___", 1:9)
    if ("chrdemo_racial_back" %in% vars && !any(vars %in% race_items)) {
      warning(
        "⚠️ You have selected ‘chrdemo_racial_back’ in your dictionary, ",
        "but none of the individual indicators (chrdemo_racial_back___1 through ___9) ",
        "were selected. The aggregated 'chrdemo_racial_back' column is created here and need the individual items. chrdemo_racial_back will therefore be empty."
      )
    }
    
    if(any(grepl("^chrdemo_racial_back___\\d", names(T)))) {
      T$chrdemo_racial_back <- NA
      for (i in 1:9) {
        col_name <- paste0("chrdemo_racial_back___", i)
        if (col_name %in% names(T)) {
          T$chrdemo_racial_back[T[[col_name]] == 1] <- i
        }
      }
    }
    if("chrdemo_working" %in% names(T) && is.numeric(T$chrdemo_working)) {
      T$chrdemo_working[T$chrdemo_working == 2] <- 1
    }
    
    
    
    ## 5.h) Exclude cases with no data except subjectkey & group----
    non_key_cols <- setdiff(names(T), c("src_subject_id", "group"))
    T <- T[rowSums(!is.na(T[, non_key_cols])) > 0, ]
    
    
    ## 5.i) Site column----
    subject_id_mapping <- data.frame(
      Abbreviation = c("BI","BM","CA","CG","CM","CP","GA","GW","HA","HK",
                       "IR","JE","KC","LA","LS","MA","ME","MT","MU","NC",
                       "NL","NN","OH","OR","PA","PI","PV","SD","SF","SG",
                       "SH","SI","SL","ST","TE","UR","WU","YA"),
      FullName = c("Beth_Israel_(Harvard)","Birmingham","Calgary_(Canada)","Cologne",
                   "Cambridge_(UK)","Copenhagen","Georgia","Gwangju",
                   "Hartford_(Institute_of_Living)","Hong_Kong",
                   "UC_Irvine","Jena","King's_College_(UK)","UCLA","Lausanne",
                   "Madrid_(Spain)","Melbourne","Montreal_(Canada)","Munich_(Germany)","UNC",
                   "Northwell","Northwestern","Ohio","Oregon","Uni_of_Pennsylvania",
                   "Pittsburgh_(UPMC)","Pavia_(Italy)","UCSD","UCSF","Singapore",
                   "Shanghai_(China)","Mt._Sinai","Seoul_(South_Korea)","Santiago",
                   "Temple","Uni_of_Rochester","Washington_University","Yale"),
      stringsAsFactors = FALSE
    )
    T <- T %>%
      mutate(Tempstore = substr(src_subject_id, 1, 2)) %>%
      left_join(subject_id_mapping, by = c("Tempstore" = "Abbreviation")) %>%
      rename(Site = FullName) %>%
      select(-Tempstore)
    
    ## 5.j) Create binary group----
    T$group <- NA
    T$group[T$phenotype == "CHR"] <- 1
    T$group[T$phenotype == "HC"]  <- 0
    
    ## remove rows with no group
    T <- T[!is.na(T$group), ]
    
    
    
    ## 5.k) Recode data----
    # Loop over every column in data and apply recode_if_present
    for (col in names(T)) {
      if (length(na.omit(T[[col]]))>0){
        T <- recode_if_present(T, col)
      }
    }  
    
    
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## 5.l) Remove columns that have no data (all NA or all "")----
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    empty_cols <- sapply(T, function(col) {
      if (is.character(col)) {
        all(is.na(col) | col == "")
      } else if (is.factor(col)) {
        # Convert factor to character for the comparison
        all(is.na(col) | as.character(col) == "")
      } else {
        all(is.na(col))
      }
    })
    
    if (any(empty_cols)) {
      removed_vars <- names(T)[empty_cols]
      T <- T[, !empty_cols, drop = FALSE]
      warning(
        "The following variables had no data at all and have been removed from the basetable: ",
        paste(removed_vars, collapse = ", ")
      )
    }
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## 5.m) Change datatype based on notes in dictionary----
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the dictionary
    dict <- readxl::read_excel(dictionary_DIR, 
                               col_types = "text")
    ignore_keys <- c(-900, -300, -997, -998)

    # Infer datatype inline: if a note contains at least two mapping pairs with keys other than -900 or -300, mark it as Categorical; otherwise, continuous.
    dict$InferredDataType <- sapply(dict$Notes, function(note) {
      pairs <- str_extract_all(note, "(-?\\d+)\\s*=\\s*[^,;]+")[[1]]
      if (length(pairs) == 0) return(NA)
      keys <- as.numeric(str_extract(pairs, "-?\\d+"))
      valid_keys <- unique(keys[!(keys %in% ignore_keys)])
      if (length(valid_keys) >= 2) "Categorical" else NA
    })
    
    # Identify variables marked as "Categorical"
    cat_vars <- dict$ElementName[dict$InferredDataType == "Categorical"]
    
    # Convert the corresponding columns in T to character
    cols_to_convert <- names(T) %in% cat_vars
    T[cols_to_convert] <- lapply(T[cols_to_convert], as.character)
    
    
    ## Change all "Not assessed" to NA
    # — replace all “not assessed” with NA across the entire T data.frame
    T[] <- lapply(T, function(col) {
      # character columns
      if (is.character(col)) {
        col[tolower(col) == "not assessed"] <- NA
        return(col)
      }
      # factor columns
      if (is.factor(col)) {
        ch <- as.character(col)
        ch[tolower(ch) == "not assessed"] <- NA
        # rebuild factor (dropping the “not assessed” level)
        return(factor(ch))
      }
      # everything else left untouched
      col
    })
    
    
    # ──────────────────────────────────────────────────────────────────────────────
    # 5.n) STANDARDISE TEMPERATURE, HEIGHT, WEIGHT UNITS
    # ──────────────────────────────────────────────────────────────────────────────
    
    # 1) Body temperature → Celsius
    if (all(c("chrchs_bodytemp","chrchs_bodytempunits") %in% names(T))) {
      u <- tolower(T$chrchs_bodytempunits)
      # detect Fahrenheit
      isF <- grepl("fah|°f|fahrenheit", u)
      # convert only those rows
      T$chrchs_bodytemp[isF] <- (T$chrchs_bodytemp[isF] - 32) * 5/9
      # now mark every row as Celsius
      T$chrchs_bodytempunits <- "Celsius"
    }
    
    # 2) Height → Centimeters
    if (all(c("chrchs_height","chrchs_heightunits") %in% names(T))) {
      u <- tolower(T$chrchs_heightunits)
      # detect inches
      isIn <- grepl("inch|inches|\\bin\\b", u)
      T$chrchs_height[isIn] <- T$chrchs_height[isIn] * 2.54
      T$chrchs_heightunits <- "cm"
    }
    
    # 3) Weight → Kilograms
    if (all(c("chrchs_weight","chrchs_weightunits") %in% names(T))) {
      u <- tolower(T$chrchs_weightunits)
      # detect pounds
      isLb <- grepl("lb|pound", u)
      T$chrchs_weight[isLb] <- T$chrchs_weight[isLb] * 0.45359237
      T$chrchs_weightunits <- "kg"
    }
    
    
    #### Warning about non applicable data
    
    # Check each column for the value -3 or -300 and capture the column names that contain -3 or -300
    cols_with_neg3 <- names(T)[sapply(T, function(x) any(x %in% c(-3,-7,-300), na.rm=TRUE))]
    
    # Return TRUE if any column has -3, otherwise FALSE
    any_neg3 <- length(cols_with_neg3) > 0
    
    if (any_neg3){
      options(warning.length = 8170) # How many variables it will show
      warning(
        paste(
          "-3, -7 and/or -300 was found in the data. -3, -7 and -300 are entered when the question was non applicable.",
          "For example, a question was skipped because of a previous question.",
          "Please look into your data and consider how to deal with these -3, -7 and/or -300 values.",
          "These variables contain -3/-7/-300:",
          paste(cols_with_neg3, collapse = ", "),
          sep = "\n"
        )
      )
    }
    
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## 5.n) Write out final table----
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if (timepoint == "baseline"){
      outdir <- file.path(data_DIR, "BASELINE_TABLES")
    } else if (timepoint == "month_1"){
      outdir <- file.path(data_DIR, "MONTH1_TABLES")
    } else if (timepoint == "month_2"){
      outdir <- file.path(data_DIR, "MONTH2_TABLES")
    } else if (timepoint == "screening"){
      outdir <- file.path(data_DIR, "SCREENING_TABLES")
    }
    
    dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
    outfile <- file.path(outdir, paste0("basetable_", Sys.Date(), ".csv"))
    outfile_meta <- file.path(outdir, paste0("metatable_", Sys.Date(), ".csv"))
    write_csv(T, outfile)
    write_csv(meta_table, outfile_meta)
    
    message("\nBasetable saved to: ", outfile)
    
    return(list(data=T, meta=meta_table))
    
  } else {
    stop("Could not detect Prescient or AMP-SCZ structure in the specified directory.")
  }
}