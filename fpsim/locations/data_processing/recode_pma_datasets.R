################################################################################
# -Recode Household/Female survey datasets from PMA to
# produce fpsim empowerment variables.
#
# -You can request PMA datasets at: https://datalab.pmadata.org/:
#
# Original analysis by Marita Zimmermann, March 2023
###############################################################################

library(tidyverse)
library(haven)
library(survey)


home_dir <-
  path.expand("~")   # replace with your own path to the DTA file
pma_dir <- "PMA"  # Replace with your own data directory structure
survey_dir <-
  "Kenya"  # Replace with your own data directory structure
file1 <-
  "PMA2019_KEP1_HQFQ_v3.0_21Oct2021/PMA2019_KEP1_HQFQ_v3.0_21Oct2022.dta"
file2 <-
  "PMA2020_KEP2_HQFQ_v3.0_21Oct2022/PMA2021_KEP2_HQFQ_v3.0_21Oct2022.dta"
file3 <-
  "PMA2022_KEP3_HQFQ_v4.0_2Jul2023/PMA2022_KEP3_HQFQ_v4.0_12Jul2023.dta"

file1_path <- file.path(home_dir, pma_dir, survey_dir, file1)
file2_path <- file.path(home_dir, pma_dir, survey_dir, file2)
file3_path <- file.path(home_dir, pma_dir, survey_dir, file3)

# Load multiple datasets and add a column wave to each of them
data1.raw <- read_dta(file1_path)
data1.raw <-
  data1.raw %>% mutate(wave = 1,
                       RE_ID = as.character(RE_ID),
                       county = as.character(county))

data2.raw <- read_dta(file2_path)
data2.raw <-
  data2.raw %>% mutate(wave = 2,
                       RE_ID = as.character(RE_ID),
                       county = as.character(county))

data3.raw <- read_dta(file3_path)
data3.raw <-
  data3.raw %>% mutate(
    wave = 3,
    RE_ID = as.character(doi_corrected),
    county = as.character(county)
  )

# Join all datasets from multiple waves -- I'm getting warnings
data.raw <- bind_rows(data1.raw, data2.raw, data3.raw)


# -- Data recode
recoded.datasets <- data.raw %>%
  # All women sample
  filter((HHQ_result == 1 |
            HHQ_result_cc == 1) &
           (FRS_result == 1 | 
              FRS_result_cc == 1) & last_night == 1 & !is.na(FQweight)
  ) %>%
  filter(female_ID != "") %>%
  
  # Select relevant variables
  select(
    wave,
    EC,
    EA_ID,
    FQdoi_corrected,
    FQdoi_correctedSIF,
    FQmarital_status,
    FQweight,
    IUD,
    LAM,
    activity_30d,
    age = FQ_age,
    birth_events,
    buy_decision_clothes,
    buy_decision_daily,
    buy_decision_major,
    buy_decision_medical,
    current_contra = current_user,
    decide_spending_mine,
    decide_spending_partner,
    diaphragm,
    ever_birth,
    female_ID,
    femalecondoms,
    femalester,
    financial_goal_yn,
    foamjelly,
    fp_ever_user,
    future_user_not_current,
    future_user_pregnant,
    highest_grade,
    implant,
    injectables,
    malecondoms,
    malester,
    mar_decision,
    mobile_money_yn,
    money_knowledge,
    money_knowledge_where_yn,
    months_pregnant,
    more_children,
    more_children_pregnant,
    othertrad,
    own_land_yn,
    partner_overall,
    pill,
    pregnant,
    recent_birth,
    recent_birthSIF,
    reliant_finance,
    rhythm,
    savings_yn,
    school,
    school_12m_yn,
    school_left_age,
    state = country,
    starts_with("activity_30d"),
    starts_with("wait_birth"),
    strata,
    stndrddays,
    ur,
    wealthquintile,
    who_earns_more,
    why_not_decision,
    withdrawal,
    work_12mo,
    work_7d,
    work_paid,
    wge_fp_eff_confident_switch,
    wge_fp_eff_discuss_fp,
    wge_fp_eff_switch,
    wge_preg_eff_decide_another,
    wge_preg_eff_decide_start,
    wge_preg_eff_decide_start_none,
    wge_preg_eff_nego_stop,
    wge_preg_eff_nego_stop_none,
    wge_preg_eff_ptnr_disc_start,
    wge_sex_aut_avoid,
    wge_sex_aut_force,
    wge_sex_aut_hurt,
    wge_sex_aut_promiscuous,
    wge_sex_aut_stop_support,
    wge_sex_eff_avoid,
    wge_sex_eff_confident,
    wge_sex_eff_decide,
    wge_sex_eff_tell_no
  ) %>%
  
  # Replace missing with NA
  mutate_at(
    vars(
      "current_contra",
      "work_paid",
      "work_12mo",
      "work_7d",
      "financial_goal_yn",
      "ever_birth",
      "fp_ever_user",
      "birth_events",
      "savings_yn",
      "money_knowledge",
      "money_knowledge_where_yn",
      "who_earns_more",
      "mar_decision",
      "partner_overall",
      "why_not_decision",
      "buy_decision_major",
      "buy_decision_daily",
      "buy_decision_medical",
      "buy_decision_clothes",
      "decide_spending_mine",
      "decide_spending_partner",
      "mobile_money_yn",
      "more_children",
      "more_children_pregnant",
      (starts_with("wge_"))
    ),
    list(~ ifelse(. == -99 | . == 96 | . == -88, NA, .))
  ) %>%
  
  # demographics
  mutate(
    date = parse_date_time(FQdoi_correctedSIF, "Y m d H M S"),
    age_grp = cut(age, c(0, 18, 20, 25, 35, 50)),
    recent_birth_date = as.Date(recent_birthSIF),
    pp.time = as.numeric(difftime(date, recent_birth_date, units = "weeks")) /
      4.333333,
    # time postpartum in months
    married = ifelse(married_cat == 1, 1, 0),
    yrs_school = factor(
      school,
      levels = c(0, 1, 2, 3, 4, 5),
      labels = c(
        "Never Attended",
        "Primary",
        "Post-Primary/Vocational",
        "Secondary/'A' Level",
        "College (Middle Level)",
        "University"
      )
    ),
    live_births = ifelse(ever_birth == 0, 0, birth_events),
    urban = ifelse(ur == 1, 1, 0),
    edu_cat = ordered(school),
    
    # years of education
    yrs.edu = case_when(
      school == 0 ~ 0,
      # never
      school == 1 ~ 0,
      # Primary, 8 years
      school == 2 ~ 8,
      # post-primary
      school == 3 ~ 8,
      # secondary, 4 years
      school == 4 ~ 8 + 4,
      # college
      school == 5 ~ 8 + 4
    ) + # university
      ifelse(highest_grade > 0 &
               highest_grade < 9, highest_grade, 0),
    # add plus year within that level
    
    # Work
    anywork_12m = ifelse(work_7d == 1, 1, work_12mo),
    paidw_12m = case_when(work_paid %in% c(1, 2, 3) ~ 1, anywork_12m == 0 |
                            work_paid == 4 ~ 0),
    who_earns_more = ifelse(paidw_12m == 0, 2, who_earns_more),
    # If no paid work, specify that partner earns more
    
    # contraception
    method_recode = ifelse(
      femalester == 1,
      "BTL/vasectomy",
      ifelse(
        malester == 1,
        "BTL/vasectomy",
        ifelse(implant == 1, "implant",
               ifelse(
                 IUD == 1, "IUD",
                 ifelse(
                   pill == 1,
                   "pill",
                   ifelse(
                     injectables == 1,
                     "injectables",
                     ifelse(
                       EC == 1,
                       "other modern",
                       ifelse(
                         malecondoms == 1,
                         "condoms",
                         ifelse(
                           femalecondoms == 1,
                           "condoms",
                           ifelse(
                             diaphragm == 1,
                             "other modern",
                             ifelse(
                               foamjelly == 1,
                               "other modern",
                               ifelse(
                                 stndrddays == 1,
                                 "other modern",
                                 ifelse(
                                   LAM == 1,
                                   "other modern",
                                   ifelse(
                                     withdrawal == 1,
                                     "withdrawal",
                                     ifelse(
                                       rhythm == 1,
                                       "other traditional",
                                       ifelse(othertrad == 1,
                                              "other traditional", NA)
                                     )
                                   )
                                 )
                               )
                             )
                           )
                         )
                       )
                     )
                   )
                 )
               ))
      )
    ),
    # method categorization via Festin et al. 2016
    # long (or permamnent) vs short (or medium) acting
    method.long  = ifelse(femalester == 1 |
                            malester == 1 |
                            implant == 1 |
                            IUD == 1, 1, 0),
    # yes/no long acting method (longer than 3 mo)
    method.short = ifelse(
      EC == 1 |
        malecondoms == 1 |
        femalecondoms == 1 |
        diaphragm == 1 |
        foamjelly == 1 |
        pill == 1 |
        injectables == 1 |
        stndrddays == 1 | LAM == 1,
      1,
      0
    ),
    # yes/no short acting method
    method.longshort = case_when(method.long == 1 ~ "Long", 
                                 method.short == 1 ~ "Short", 
                                 .default = "None"),
    # modern vs traditional
    method.trad  = ifelse(withdrawal == 1 |
                            rhythm == 1 | othertrad == 1, 1, 0),
    # yes/no traditional method
    method.mod  = ifelse(
      femalester == 1 |
        malester == 1 |
        implant == 1 |
        IUD == 1 |
        injectables == 1 |
        EC == 1 |
        malecondoms == 1 |
        femalecondoms == 1 |
        diaphragm == 1 |
        foamjelly == 1 |
        pill == 1 |
        stndrddays == 1 | LAM == 1,
      1,
      0
    ),
    # yes/no modern method
    method.tradmod = case_when(
      method.trad == 1 ~ "Traditional",
      method.mod == 1 ~ "Modern",
      .default = "None"
    ),
    # hormonal vs non
    method.hormonal = ifelse(
      implant == 1 |
        IUD == 1 |
        pill == 1 | injectables == 1 | EC == 1 | foamjelly == 1,
      1,
      0
    ),
    method.nonhormonal = ifelse(
      femalester == 1 |
        malester == 1 |
        malecondoms == 1 |
        femalecondoms == 1 |
        diaphragm == 1 |
        stndrddays == 1 | LAM == 1 | withdrawal == 1 | rhythm == 1,
      1,
      0
    ),
    method.hormnon = case_when(
      method.hormonal == 1 ~ "Hormonal",
      method.nonhormonal == 1 ~ "Non-hormonal",
      .default = "None"
    ),
    # Indicator for intention to use contraception (assumed yes if currently using)
    intent_contra = case_when(
      future_user_not_current == 1 |
        future_user_pregnant == 1 | current_contra == 1 ~ 1,
      TRUE ~ 0
    ),
    # Indicator for who decides about using contraception: 1 her, 2 partner, 3 joint, 96 other
    FP_who = ifelse(is.na(why_not_decision), partner_overall, why_not_decision),
    
    # intent to use contraception
    intent_cat = factor(ifelse(
      current_contra == 1,
      "User",
      ifelse(intent_contra == 1, "intent", "no_intent")
    )),
    # create factor variable for intent to use contraception
    
    # fertility intention in next 12 months
    fertility_intent = case_when(
      more_children == 0 |
        more_children_pregnant == 0 ~ 0,
      # Do not want more children
      wait_birth == 1 &
        wait_birth_value <= 12 ~ 1,
      # says want to wait 12 moths or less
      wait_birth == 1 &
        wait_birth_value > 12 ~ 0,
      # says want to more than 12 months
      wait_birth_pregnant == 1 &
        wait_birth_value + 9 - months_pregnant <= 12 ~ 1,
      # less that 12 months including current pregnancy time
      wait_birth == 2 |
        wait_birth_pregnant == 2 ~ 0,
      # says want to wait 1 year or more
      wait_birth == 3 |
        wait_birth_pregnant == 3 ~ 1,
      # says want to get pregnant soon/now
      wait_birth == 1 &
        wait_birth_value > 12 ~ 0,
      # says want to more than 12 months
      wait_birth_pregnant == 1 &
        wait_birth_value + 9 - months_pregnant > 12 ~ 0
    ),
    # more than 12 months including current pregnancy time
    
    # Empowerment
    wge_preg_eff_start = ifelse(
      is.na(wge_preg_eff_decide_start_none),
      wge_preg_eff_decide_start,
      wge_preg_eff_decide_start_none
    ),
    wge_preg_eff_stop = ifelse(
      is.na(wge_preg_eff_nego_stop_none),
      wge_preg_eff_nego_stop,
      wge_preg_eff_nego_stop_none
    ),
    decide_spending_mine_alone = case_when(
      paidw_12m == 0 ~ 0,
      # Version of below with woman alone
      decide_spending_mine %in% c(1) ~ 1,
      decide_spending_mine %in% c(2, 3, 96) ~ 0
    ),
    decide_spending_partner_alone = case_when(
      decide_spending_partner %in% c(1) ~ 1,
      # Version of below with woman alone
      decide_spending_partner %in% c(2, 3, 96) ~ 0
    ),
    decide_spending_mine =     case_when(
      paidw_12m == 0 ~ 0,
      # Who usually makes decisions about how your earnings will be used: you, your husband/partner, you and your husband/partner jointly, or someone else?
      decide_spending_mine %in% c(1, 3) ~ 1,
      decide_spending_mine %in% c(2, 96) ~ 0
    ),
    decide_spending_partner =  case_when(
      decide_spending_partner %in% c(1, 3) ~ 1,
      # Who usually makes decisions about how your husband/partner's earnings will be used: you, your husband/partner, you and your husband/partner jointly, or someone else?
      decide_spending_partner %in% c(2, 96) ~ 0
    ),
    wge_sex_eff_confident =    case_when(
      wge_sex_eff_confident %in% c(1, 2, 3) ~ 0,
      # I am confident I can tell my husband/partner when I want to have sex.
      wge_sex_eff_confident %in% c(4, 5) ~ 1
    ),
    wge_sex_eff_tell_no =      case_when(
      wge_sex_eff_tell_no %in% c(1, 2, 3) ~ 0,
      # If I do not want to have sex, I can tell my husband/partner.
      wge_sex_eff_tell_no %in% c(4, 5) ~ 1
    ),
    wge_sex_eff_decide =       case_when(
      wge_sex_eff_decide %in% c(1, 2, 3) ~ 0,
      # I am able to decide when to have sex. unfortunately not asked wave 2
      wge_sex_eff_decide %in% c(4, 5) ~ 1
    ),
    wge_preg_eff_start =       case_when(
      wge_preg_eff_start %in% c(1, 2, 3) ~ 0,
      # I can/could decide when I want to start having children.
      wge_preg_eff_start %in% c(4, 5) ~ 1
    ),
    wge_preg_eff_stop =        case_when(
      wge_preg_eff_stop %in% c(1, 2, 3) ~ 0,
      # I will be able to negotiate with my husband/partner when to stop having children. OR  I can negotiate with my husband/partner when to stop having children.
      wge_preg_eff_stop %in% c(4, 5) ~ 1
    ),
    buy_decision_health =      case_when(
      buy_decision_medical %in% c(1, 3) ~ 1,
      # Who usually makes decisions about getting medical treatment for yourself: you, your husband/partner, you and your husband/partner jointly, or someone else?
      buy_decision_medical %in% c(2, 96) ~ 0
    ),
    buy_decision_major =       case_when(
      buy_decision_major %in% c(1, 3) ~ 1,
      # Who usually makes decisions about making large household purchases: you, your husband/partner, you and your husband/partner jointly, or someone else?
      buy_decision_major %in% c(2, 96) ~ 0
    ),
    buy_decision_daily =       case_when(
      buy_decision_daily %in% c(1, 3) ~ 1,
      # Who usually makes decisions about making household purchases for daily needs: you, your husband/partner, you and your husband/partner jointly, or someone else?
      buy_decision_daily %in% c(2, 96) ~ 0
    )
  ) %>%
  
  # create variable for years since first observation
  group_by(female_ID) %>% 
  arrange(wave) %>%
  mutate(
    start_date = min(date, na.rm = TRUE),
    years = interval(start_date, date) %/% months(1) / 12
  )

rm(data.raw, data1.raw, data2.raw, data3.raw)
