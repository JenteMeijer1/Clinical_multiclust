#' Create a Descriptive Table for AMP_SCZ or Prescient data
#'
#' This function generates a summary table for selected variables, automatically
#' converting categorical variables to factors when there are less than 5 unique numbers. 
#'
#' @param df A df frame containing the dfset.
#' @param dictionary_DIR A directory to where the dictionary is saved.
#' @param comparison A character that indicates which variable you want to use to compare between groups. 
#'                   Default is `group`
#' @param comparison_labels A named character vector specifying the labels for the comparison variable.
#'                          Names should match the values in `comparison`, and values should be human-readable group labels.
#'                          Default is `c("1" = "CHR", "0" = "HC")`. For Prescient this is: `c("UHR" = "CHR", "HealthyControl" = "HC")`
#'
#' @return A formatted summary table as a `flextable` object, which includes:
#'         - Descriptive statistics (mean, SD for continuous; proportions for categorical)
#'         - Statistical test results (t-tests for continuous, chi-square for categorical, Fisher’s exact for dichotomous)
#'         - False Discovery Rate (FDR) adjusted p-values
#'         - Effect sizes (Cohen's d for continuous variables, Cramér’s V for categorical variables)
#' @details
#' This function automatically performs the following checks and processing:
#' - Ensures `df` is a valid data frame.
#' - Selects only variables that exist in `df` and removes those that are completely missing.
#' - Converts categorical variables with ≤5 unique values into factors.
#' - Applies group labels to the `comparison` variable if `comparison_labels` is provided.
#' - Computes p-values and adjusts them for multiple comparisons using FDR correction.
#' - Computes effect sizes and ensures they appear only in the main statistics row.
#' @import gtsummary dplyr forcats flextable effectsize
#' @export
#'
#' @examples
#' # Example Usage:
#' data <- read.csv("your_data.csv")
#' dictionary_DIR <- "~/Library/CloudStorage/OneDrive-Personal/OneDrive/PhD computational mental health/Code/GIT/Docs/1Complete_dictionary.xlsx"
#' comparison <- "group"
#' comparison_labels <- "c("1" = "CHR", "0" = "HC")"
#' table <- demographic_table(data, dictionary_DIR, comparison_labels) 
#'

demographic_table <- function(df, dictionary_DIR, comparison, comparison_labels= c()) {
  library(gtsummary)
  library(dplyr)
  library(forcats)
  library(flextable)
  library(readr)
  library(stringr)
  library(tidyr)
  library(readxl)
  library(effectsize)  # For effect size calculations
  
  # Custom warning function for improved readability
  print_warning <- function(warning_text) {
    if (length(warning_text) > 0) {
      cat("\n⚠ Warning:\n", paste(warning_text, collapse = "\n"), "\n\n")
    }
  }
  
  ### Error Handling and Validations ----
  if (!is.data.frame(df)) stop("Error: 'df' must be a data frame.")
  if (nrow(df) == 0) stop("Error: The dataset is empty (0 rows). Please provide a valid dataset.")

  
  #'##########################################################################################################
  ## Load necessary functions ----
  #'##########################################################################################################
  
  variable_selection <- function(dictionary_DIR, required_vars = c(comparison)) {
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
    
    #Remove duplicates
    selected_vars <- unique(selected_vars)
    
    # Select variables for the demographics
    selected_vars_demographics <- dict %>%
      filter(Include_demographics %in% c("Yes", "yes")) %>%
      select(ElementName, Label)
    
    #Remove duplicates
    selected_vars_demographics <- unique(selected_vars_demographics)
    
    # Create metatable with info about modality
    metatable <- dict %>%
      filter(Include_basetable %in% c("Yes", "yes")) %>%
      select(ElementName, Datatype)
    
    # Ensure required_vars are included
    missing_required <- setdiff(required_vars, selected_vars$ElementName)
    if (length(missing_required) > 0) {
      selected_vars <- bind_rows(
        selected_vars,
        tibble(ElementName = missing_required, Label = missing_required)
      )
    }
    
    # Ensure required_vars are included
    missing_required <- setdiff(required_vars, selected_vars_demographics$ElementName)
    if (length(missing_required) > 0) {
      selected_vars_demographics <- bind_rows(
        selected_vars_demographics,
        tibble(ElementName = missing_required, Label = missing_required)
      )
    }
    
    # Capture the idctionary's order of variables
    dict_order <- dict$ElementName
    
    
    # Extract final sets
    vars_basetable    <- selected_vars$ElementName
    vars_demographics <- selected_vars_demographics$ElementName
    labels            <- setNames(as.list(selected_vars$Label), selected_vars$ElementName)
    
    return(list(vars_basetable, vars_demographics, labels, dict_order))
  }
  
  #'##########################################################################################################
  
  ## Variable selection ----
  selected_variables <- variable_selection(dictionary_DIR, required_vars = c(comparison))
  variables <- selected_variables[[2]]
  labels <- selected_variables[[3]]
  
  
  # Select variables that exist in the dataset
  existing_vars <- variables[variables %in% colnames(df)]
  
  # Identify and warn about missing variables
  missing_vars <- setdiff(variables, existing_vars)
  print_warning(paste("The following variables do not exist in the dataset and will be skipped:", 
                      paste(missing_vars, collapse = ", ")))
  
  # Remove variables that are completely NA
  valid_vars <- existing_vars[sapply(df[existing_vars], function(x) !all(is.na(x)))]
  
  # Identify and warn about variables that contain only NA values
  na_vars <- setdiff(existing_vars, valid_vars)
  print_warning(paste("The following variables contain only NA values and will be skipped:", 
                      paste(na_vars, collapse = ", ")))
  
  # Stop execution if no valid variables remain
  if (length(valid_vars) == 0) stop("Error: None of the specified 'variables' exist or contain data.")
  
  # Ensure labels match only valid variables
  valid_labels <- labels[names(labels) %in% valid_vars]
  
  # Convert comparison variable to character if needed
  if (!is.factor(df[[comparison]])) {
    print_warning(paste("The comparison variable", comparison, "is not a character Converting to factor."))
    df[[comparison]] <- as.character(df[[comparison]])
  }
  
  # Apply comparison labels if provided
  if (!is.null(comparison_labels)) {
    df[[comparison]] <- factor(df[[comparison]], levels = names(comparison_labels), labels = comparison_labels)
  }
  
  # Convert variables to factors when there are small unique values
  df <- df %>%
    mutate(across(all_of(valid_vars), ~ {
      if (is.numeric(.) && n_distinct(na.omit(.)) <= 5) {
        factor(.)  # Convert to factor, preserving original values
      } else {
        .
      }
    }))
  
  ## Remove rows with NA groups
  df <- df[!is.na(df[[comparison]]), ]
  
  
  # Find number of groups
  ngroups = length(levels(df[[comparison]]))
  if (ngroups <1){
    ngroups = length(unique(df[[comparison]]))
  }
  
  ### **Check for Valid Statistical Tests with Sufficient Data in Both Groups** ###
  group_counts <- numeric(ngroups)
  valid_tests <- valid_vars[sapply(valid_vars, function(var) {
    for (i in 1:ngroups){
      group_counts[i] = length(df[[var]][df[[comparison]]==levels(df[[comparison]])[i]])
      if (group_counts[i] < 1){
        group_counts[i] = length(df[[var]][df[[comparison]]==unique(df[[comparison]])[i]])
      }
    }
    
    return(length(group_counts) > 1 && group_counts[1] > 30 && group_counts[2] >30)
  }
  )]
  
  
  # Warn and remove variables that cannot be tested
  invalid_tests <- setdiff(valid_vars, valid_tests)
  print_warning(paste("The following variables could not be tested due to lack of data in one or more groups and will be skipped:", 
                      paste(invalid_tests, collapse = ", ")))
  
  # Update valid variables to only include ones that passed the check
  valid_vars <- valid_tests
  
  ## Compute Effect Sizes Only for Valid Variables ----
  effect_sizes <- data.frame(variable = valid_vars, effect_size = NA)
  for (var in valid_vars) {
    if (is.numeric(df[[var]])) {
      if (ngroups == 2) {
        effect_sizes$effect_size[effect_sizes$variable == var] <-
          format(round(cohens_d(df[[var]] ~ df[[comparison]])$Cohens_d, 2), nsmall = 2)
      } else {
        # For more than 2 groups, compute eta squared from a one-way ANOVA
        aov_model <- aov(df[[var]] ~ df[[comparison]], data = df)
        aov_summary <- summary(aov_model)[[1]]
        ss_between <- aov_summary$`Sum Sq`[1]
        ss_total <- sum(aov_summary$`Sum Sq`)
        eta2 <- ss_between / ss_total
        effect_sizes$effect_size[effect_sizes$variable == var] <-
          format(round(eta2, 2), nsmall = 2)
      }
    } else if (is.factor(df[[var]])) {
      effect_sizes$effect_size[effect_sizes$variable == var] <-
        format(round(cramers_v(df[[var]], df[[comparison]])$Cramers_v, 2), nsmall = 2)
    } else if (is.character(df[[var]])) {
      effect_sizes$effect_size[effect_sizes$variable == var] <-
        format(round(cramers_v(df[[var]], df[[comparison]])$Cramers_v, 2), nsmall = 2)
    }
  }

  ## Reorder ----
  # Reorder variables based on dictionary
  dict_order <- selected_variables[[4]]
  final_vars <- valid_vars[order(match(valid_vars,dict_order))]
  
  
  # Use t.test when 2 groups and otherwise anove
  ## Choose statistical test for continuous variables based on number of groups ----
  if (ngroups == 2) {
    test_cont <- "t.test"
  } else {
    test_cont <- "oneway.test"
  }
  
  ## Generate Table with Only Valid Variables ----
  table <- df %>%
    select(all_of(final_vars)) %>%
    tbl_summary(
      by = comparison,
      type = all_continuous() ~ "continuous2",
      statistic = all_continuous() ~ c("{mean} ({sd})"),
      digits = list(interview_age ~ c(1, 2)),
      missing = "no",
      label = valid_labels,
      sort = list(all_categorical() ~ "frequency")
    ) %>%
    add_p(
      test = list(
        all_continuous() ~ test_cont, #TODO add anova for multiple groups
        all_categorical() ~ "chisq.test",
        all_dichotomous() ~ "fisher.test"
      ),
      test.args = list(
        all_dichotomous() ~ list(workspace = 2e6)
      )
    ) %>%
    add_overall() %>%  # Move add_overall() here so that variable labels are preserved.
    modify_table_body(
      ~ .x %>%
        # Use relationship = "many-to-many" to silence the join warning (if that is intended)
        left_join(effect_sizes, by = c("variable" = "variable"), relationship = "many-to-many") %>%
        group_by(variable) %>% 
        mutate(effect_size = ifelse(row_number() == 1, effect_size, NA)) %>%  # Keep only in main row
        ungroup() %>%
        mutate(
          p.adjusted = p.adjust(p.value, method = "fdr"),
          p.adjusted = ifelse(
            as.numeric(p.adjusted) < 0.001, "<0.001",
            ifelse(
              as.numeric(p.adjusted) > 0.99, ">0.99",
              format(round(as.numeric(p.adjusted), 3), nsmall = 3)
            )
          )
        )
    ) %>%
    modify_header(
      p.adjusted = "**FDR-Adjusted p-Value**",
      effect_size = "**Effect Size**"
    ) %>%
    modify_table_styling(
      columns = c("p.adjusted", "effect_size"),
      rows = p.adjusted == "<0.001" | as.numeric(gsub("<|>", "", p.adjusted, fixed = TRUE)) < 0.05,
      text_format = "bold"
    ) %>%
    bold_p(t = 0.05) %>%
    modify_table_body(~ .x %>% relocate(effect_size, .after = last_col())) %>%  # Move effect_size to last column
    modify_footnote(
      effect_size = "Effect Size: Cohen’s d for continuous variables, and Cramér’s V for categorical variables."
    ) %>%
    as_flex_table()
  
  return(table)
}
