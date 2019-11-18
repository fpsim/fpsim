#!/usr/bin/python

import os
from CampaignClass import *
from CampaignEnum import *
import pylab as pl

# -----------------------------------------------------------------------------
# --- HELPER FUNCTIONS - Could become part of core code
# -----------------------------------------------------------------------------

# Distribute the intervention on this day to target population

def distributeIntervention( Start_Day=0.0,
                            Nodeset_Config=None,
                            Target_Demographic="Everyone",
                            Target_Gender="All",
                            Target_Age_Min=0.0,
                            Target_Age_Max=9.3228e+35,
                            Property_Restrictions_Within_Node=[],
                            Node_Property_Restrictions=[],
                            Demographic_Coverage=1.0,
                            Intervention_Config=None ):

    if( Nodeset_Config == None ):
        Nodeset_Config = NodeSetAll()

    ce = CampaignEvent(
             Start_Day=Start_Day,
             Nodeset_Config=Nodeset_Config,
             Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                 Demographic_Coverage=Demographic_Coverage,
                 Target_Demographic=Target_Demographic,
                 Target_Gender=Target_Gender,
                 Target_Age_Min=Target_Age_Min,
                 Target_Age_Max=Target_Age_Max,
                 Property_Restrictions_Within_Node=Property_Restrictions_Within_Node,
                 Node_Property_Restrictions=Node_Property_Restrictions,
                 Intervention_Config=Intervention_Config ) )
    return ce


# Distribute the intervention when event occurs

def distributeInterventionOnEvent( Start_Day=0.0,
                                   Nodeset_Config=None,
                                   Target_Demographic="Everyone",
                                   Target_Gender="All",
                                   Target_Age_Min=0.0,
                                   Target_Age_Max=9.3228e+35,
                                   Property_Restrictions_Within_Node=[],
                                   Node_Property_Restrictions=[],
                                   Demographic_Coverage=1.0,
                                   Trigger_Condition_List=[],
                                   Intervention_Config=None ):

    if( Nodeset_Config == None ):
        Nodeset_Config = NodeSetAll()

    ce = CampaignEvent(
             Start_Day=Start_Day,
             Nodeset_Config=Nodeset_Config,
             Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                Intervention_Config=NodeLevelHealthTriggeredIV(
                    Trigger_Condition_List=Trigger_Condition_List,
                    Demographic_Coverage=Demographic_Coverage,
                    Target_Demographic=Target_Demographic,
                    Target_Gender=Target_Gender,
                    Target_Age_Min=Target_Age_Min,
                    Target_Age_Max=Target_Age_Max,
                    Property_Restrictions_Within_Node=Property_Restrictions_Within_Node,
                    Node_Property_Restrictions=Node_Property_Restrictions,
                    Actual_IndividualIntervention_Config=Intervention_Config) ) )

    return ce


# -----------------------------------------------------------------------------
# --- CONSTANTS - Constants to avoid typos
# -----------------------------------------------------------------------------

IP_UNDER_AGE   = "CurrentStatus:UNDER_AGE"
IP_PREGNANT    = "CurrentStatus:PREGNANT"
IP_POST_PARTUM = "CurrentStatus:POST_PARTUM"
IP_IMPLANT     = "CurrentStatus:IMPLANT"
IP_PILL        = "CurrentStatus:PILL"
IP_WITHDRAWAL  = "CurrentStatus:WITHDRAWAL"
IP_NONE        = "CurrentStatus:NONE"

Should_Not_Be_Broadcasted = "Should_Not_Be_Broadcasted"
Choose_Next_Method_Currently_Under_Age     = "Choose_Next_Method_Currently_Under_Age"
Choose_Next_Method_Currently_Post_Partum   = "Choose_Next_Method_Currently_Post_Partum"
Choose_Next_Method_Currently_On_Implant    = "Choose_Next_Method_Currently_On_Implant"
Choose_Next_Method_Currently_On_Pill       = "Choose_Next_Method_Currently_On_Pill"
Choose_Next_Method_Currently_On_Withdrawal = "Choose_Next_Method_Currently_On_Withdrawal"
Choose_Next_Method_Currently_On_None       = "Choose_Next_Method_Currently_On_None"

USE_IMPLANT    = "Use_Implant"
USE_PILL       = "Use_Pill"
USE_WITHDRAWAL = "Use_Withdrawal"
USE_NONE       = "Use_None"

possible_mother_min_age = 15.0
possible_mother_max_age = 50.0

# -----------------------------------------------------------------------------
# --- POPULATION INITIALIZATION - Functions for initializing the population
# ---                             with contraceptives
# -----------------------------------------------------------------------------

# Initialize women under 15 at the start of the simulation with a delay so that
# they choose a contraceptive about the time they turn 15.
# !!! COULD PROBABLY USE REFINEMENT !!!

def InitializeUnderAgeWomen( campaign, num_age_steps = 4 ):
    be_cnm_under_age = BroadcastEvent(Choose_Next_Method_Currently_Under_Age)

    age = 0.0
    age_step = possible_mother_min_age / num_age_steps
    for age in pl.arange(0, possible_mother_min_age, age_step):
        mean_delay_years = possible_mother_min_age - (age + age_step/2)
        delay = DelayedIntervention( Delay_Period_Distribution = DelayedIntervention_Delay_Period_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                     Delay_Period_Gaussian_Mean = 365.0 * mean_delay_years,
                                     Delay_Period_Gaussian_Std_Dev = 180.0,
                                     New_Property_Value = IP_UNDER_AGE,
                                     Actual_IndividualIntervention_Configs = [be_cnm_under_age] )

        ce = distributeIntervention( Target_Demographic = "ExplicitAgeRangesAndGender",
                                     Target_Gender = "Female",
                                     Target_Age_Min = age,
                                     Target_Age_Max = age+age_step,
                                     Intervention_Config = delay )
        campaign.add_campaign_event( ce )


# Initialize the possible mothers at the start of the simulation with a contraceptive.
# The quick solution is to have them choose a method as if they just turned 15.
# !!! COULD PROBABLY USE REFINEMENT !!!

def InitializePossibleMothers( campaign ):
    ce = distributeIntervention( Target_Demographic = "ExplicitAgeRangesAndGender",
                                 Target_Gender = "Female",
                                 Target_Age_Min = possible_mother_min_age,
                                 Target_Age_Max = possible_mother_max_age,
                                 Intervention_Config = BroadcastEvent(Choose_Next_Method_Currently_Under_Age) )
    campaign.add_campaign_event( ce )


def InitializePregnantWomen( campaign ):
    # TODO - IP_PREGNANT not working
    pregnant_PVC = PropertyValueChanger( Intervention_Name="Pregnant", 
        Target_Property_Key = 'CurrentStatus',
        Target_Property_Value = 'PREGNANT'
    )
    ce = distributeIntervention( Target_Demographic = "Pregnant",
                                 Intervention_Config = pregnant_PVC )
    campaign.add_campaign_event( ce )


# -----------------------------------------------------------------------------
# --- RESPOND TO BUILT-IN EVENTS - Functions for responding to built-in events
# --- such as Births, Pregnant, and GaveBirth.  These functions will be needed
# --- in most FP scenarios.
# -----------------------------------------------------------------------------

# Distribute a delay to new borns so that they choose a contraceptive about the
# time they become a possible mother.

# ??? 1) DO WE NEED A BUILT-IN EVENT SO THAT WE KNOW EXACTLY WHEN THEY BECOME
# ??? A POSSIBLE MOTHER?
# @@@ -> Have event StartBeingPossibleMother that triggers at Possible_Mother_Min_Age_Years

# ??? 2) DO WE WANT TO USE CONTRACEPTIVE INSTEAD OF DELAYED INTERVENTION?

def DistributeDelayToNewBorns( campaign ):
    be_cnm_under_age = BroadcastEvent(Choose_Next_Method_Currently_Under_Age)

    delay = DelayedIntervention( Delay_Period_Distribution = DelayedIntervention_Delay_Period_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                 Delay_Period_Gaussian_Mean = possible_mother_min_age*365.0,
                                 Delay_Period_Gaussian_Std_Dev = 180.0,
                                 New_Property_Value = IP_UNDER_AGE,
                                 Actual_IndividualIntervention_Configs = [be_cnm_under_age] )

    ce = distributeInterventionOnEvent( Target_Demographic = "ExplicitGender",
                                        Target_Gender = "Female",
                                        Trigger_Condition_List=["Births"],
                                        Intervention_Config = delay )
    campaign.add_campaign_event( ce )


# Change CurrentStatus IP to "pregnant" mainly to cause an individual's
# existing contraceptive to abort.  That is, this contraceptive will replace an
# existing one when women becomes pregnant.

def ChangeCurrentStatusToPregnant( campaign ):

    pregnant_PVC = PropertyValueChanger( Intervention_Name="Pregnant",
        Target_Property_Key = 'CurrentStatus',
        Target_Property_Value = 'PREGNANT'
    )

    ce = distributeInterventionOnEvent(Target_Demographic="ExplicitGender",
                                       Target_Gender="Female",
                                       Trigger_Condition_List=["Pregnant"],
                                       Intervention_Config=pregnant_PVC)
    campaign.add_campaign_event( ce )


# Distribute a "post partum" contraceptive to women when they give birth.

# !!! THIS WOULD BE THE FUNCTION TO MODIFY/REPLACE/OVERRIDE IF WE WANTED
# !!! SOME WOMEN TO USE AN IUD.

def DistributeContraceptiveToPostPartumMothers( campaign ):
    con_post_partum = Contraceptive( Intervention_Name="Post_Partum",
                                     New_Property_Value=IP_POST_PARTUM,
                                     Disqualifying_Properties=[ IP_PREGNANT ],
                                     Waning_Config=WaningEffectExponential(InitialEfficacy=1.0,Decay_Time_Constant=180),
                                     Usage_Duration_Distribution = Contraceptive_Usage_Duration_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                     Usage_Duration_Gaussian_Mean = 180,
                                     Usage_Duration_Gaussian_Std_Dev = 10,
                                     Usage_Expiration_Event = Choose_Next_Method_Currently_Post_Partum )

    ce = distributeInterventionOnEvent( Target_Demographic = "ExplicitGender",
                                        Target_Gender = "Female",
                                        Trigger_Condition_List=["GaveBirth"],
                                        Intervention_Config = con_post_partum )
    campaign.add_campaign_event( ce )



# -----------------------------------------------------------------------------
# --- SPECIFIC CAMPAIGN HELPER - helper functions for this campaign
# -----------------------------------------------------------------------------

def DistributeInterventionList( campaign, eventInterventionPairList ):
    for event_name, interven in eventInterventionPairList:
        ce = distributeInterventionOnEvent( Target_Demographic = "ExplicitGender",
                                            Target_Gender = "Female",
                                            Trigger_Condition_List=[event_name],
                                            Intervention_Config = interven )
        campaign.add_campaign_event( ce )


# -----------------------------------------------------------------------------
# --- CAMPAIGN GENERATION
# -----------------------------------------------------------------------------

# This is the top-level function for generating a campaign for the FP scenario.
# It should be called from within the scripts for generating a simulation run.

# !!! The inputs to this function are TBD.  It is expected that the available
# !!! Contraceptives and the RandomChoiceXXX objects for selecting them can
# !!! change from run to run.

def GenerateCampaignFP( contraceptiveList, randomChoiceList ):
    campaign = Campaign( Campaign_Name="Simple FP Campaign" )

    InitializeUnderAgeWomen( campaign )
    InitializePossibleMothers( campaign )
    InitializePregnantWomen( campaign )
    DistributeDelayToNewBorns( campaign )

    DistributeInterventionList( campaign, randomChoiceList )
    DistributeInterventionList( campaign, contraceptiveList )

    ChangeCurrentStatusToPregnant( campaign )
    DistributeContraceptiveToPostPartumMothers( campaign )

    return campaign


# -----------------------------------------------------------------------------
# --- CONTRACEPTIVES
# -----------------------------------------------------------------------------

# Each Contraceptive intervention must specify:
#    New_Property_Value = <name>                - provides data on current method
#    Disqualifying_Properties = [ IP_PREGNANT ] - aborts this method when becoming pregnant

# !!! I'm not sure what will change with the contraceptives except maybe the Usage_Duration.

# NOTE: One could still create the list of all possible contraceptives even if
#       the logic for choosing them doesn't select them.  It will just sit on the
#       node waiting for the event.

def CreateContraceptives():

    # TODO: Add other methods, e.g. IUD (hormonal and copper-T)

    # This is modeling a 3-year sub-dermal hormonal implant
    con_implant    = Contraceptive( Intervention_Name = "Implant",
                                    New_Property_Value = IP_IMPLANT,
                                    Disqualifying_Properties = [ IP_PREGNANT ],
                                    Waning_Config = WaningEffectBoxExponential(Initial_Effect=0.995, Box_Duration=3*365, Decay_Time_Constant=365),
                                    Usage_Duration_Distribution = Contraceptive_Usage_Duration_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                    Usage_Duration_Gaussian_Mean = 3*365,
                                    Usage_Duration_Gaussian_Std_Dev = 180,
                                    Usage_Expiration_Event = Choose_Next_Method_Currently_On_Implant )

# !!! Is 90 pills the size of a typical prescription, or from the DHS data we have?
    con_pill       = Contraceptive( Intervention_Name = "Pill",
                                    New_Property_Value = IP_PILL,
                                    Disqualifying_Properties =[ IP_PREGNANT ],
                                    Waning_Config = WaningEffectBox(Initial_Effect=0.9, Box_Duration=90),
                                    Usage_Duration_Distribution = Contraceptive_Usage_Duration_Distribution_Enum.UNIFORM_DISTRIBUTION,
                                    Usage_Duration_Constant = 90,
                                    Usage_Expiration_Event = Choose_Next_Method_Currently_On_Pill )

    con_withdrawal = Contraceptive( Intervention_Name = "Withdrawal",
                                    New_Property_Value = IP_WITHDRAWAL,
                                    Disqualifying_Properties = [ IP_PREGNANT ],
                                    Waning_Config = WaningEffectConstant(Initial_Effect=0.2),
                                    Usage_Duration_Distribution = Contraceptive_Usage_Duration_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                    Usage_Duration_Gaussian_Mean = 180,
                                    Usage_Duration_Gaussian_Std_Dev = 10,
                                    Usage_Expiration_Event = Choose_Next_Method_Currently_On_Withdrawal )

# !!! Why 365 mean duration for None?
    con_none       = Contraceptive( Intervention_Name = "None",
                                    New_Property_Value = IP_NONE,
                                    Disqualifying_Properties = [ IP_PREGNANT ],
                                    Waning_Config = WaningEffectConstant(Initial_Effect=0.0),
                                    Usage_Duration_Distribution = Contraceptive_Usage_Duration_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                    Usage_Duration_Gaussian_Mean = 365,
                                    Usage_Duration_Gaussian_Std_Dev = 3,
                                    Usage_Expiration_Event = Choose_Next_Method_Currently_On_None )

    con_list = []
    con_list.append( (USE_IMPLANT   , con_implant  ) )
    con_list.append( (USE_PILL      , con_pill      ) )
    con_list.append( (USE_WITHDRAWAL, con_withdrawal) )
    con_list.append( (USE_NONE      , con_none      ) )

    return con_list


# -----------------------------------------------------------------------------
# --- RANDOM CHOICES MATRIX
# -----------------------------------------------------------------------------

def CreateRandomChoiceMatrixList():

    # FYI: Each probability matrix is YEARS x AGES x PARITY in dimensions. The
    # first level is the years, second level is ages, and the rows of
    # numbers correspond to parity levels.

    # Hardcoded values for filters works as an example only.  Currently using same filters for all methods for demonstration purposes only.
    # TODO: If one of the filter is empty, we need to add "Filters": [] in the campaign file manually. Otherwise,
    # we will meet this exception since [] is the default value to Filters so dtk_tools will not save it to json:
    #   "MissingParameterFromConfigurationException:
    #     Exception in utils\Configure.cpp at 1097 in Kernel::JsonConfigurable::handleMissingParam.
    #     Parameter 'Filters of RandomChoiceMatrix' not found in input file 'campaign-RCM.json'."

    # Order of multipliers is currently 1. Implant, 2. Pill, 3. Withdrawl.  Other contraceptive methods are not used in this example.
    filters = []
    filters.append(Filter(Properties=["CurrentStatus:PILL", "Knowledge:NO"], Multipliers = [ 0, 1, 0 ]))
    filters.append(Filter(Properties=["CurrentStatus:NONE", "Knowledge:NO"], Multipliers = [ 0, 1, 2 ]))
    filters.append(Filter(Properties=["CurrentStatus:PILL", "Knowledge:YES"], Multipliers = [ 1, 1, 2 ]))
    filters.append(Filter(Properties=["CurrentStatus:NONE", "Knowledge:YES"], Multipliers = [ 1, 1, 1 ]))

    # initialize contraceptive in the population.
    pm_implant = [
        [
            [ 0.1, 0.2 ],
            [ 0.3, 0.4 ],
            [ 0.5, 0.6 ]
        ],
        [
            [ 0.3, 0.4 ],
            [ 0.5, 0.6 ],
            [ 0.7, 0.8 ]
        ],
    ]
    under_age_implant = Choice( Broadcast_Event=USE_IMPLANT,
                                 Years=[ 1990, 2000 ],
                                 Ages=[ 15, 25, 40 ],
                                 Parity=[ 0, 3 ],
                                 ProbabilityMatrix=pm_implant )

    pm_pill = [
        [
            [ 0.1, 0.2, 0.3 ],
            [ 0.5, 0.6, 0.7 ]
        ],
        [
            [ 0.7, 0.6, 0.5 ],
            [ 0.4, 0.3, 0.2 ]
        ]
    ]
    under_age_pill = Choice( Broadcast_Event=USE_PILL,
                             Years=[ 1992, 2050 ],
                             Ages=[ 20, 50 ],
                             Parity=[ 0, 2, 4 ],
                             ProbabilityMatrix=pm_pill )

    pm_withdrawal = [
        [
            [ 0.8 ]
        ],
        [
            [ 0.2 ]
        ]
    ]
    under_age_withdrawal = Choice( Broadcast_Event=USE_WITHDRAWAL,
                                   Years=[ 1992, 2050 ],
                                   Ages=[ 50 ],
                                   Parity=[ 10 ],
                                   ProbabilityMatrix=pm_withdrawal )


    choices_under_age = []
    choices_under_age.append( under_age_implant )
    choices_under_age.append( under_age_pill )
    choices_under_age.append( under_age_withdrawal )

    filters_under_age = filters

    rcm_from_under_age = RandomChoiceMatrix( Choices=choices_under_age, Filters=filters_under_age,  use_defaults=True )

    # Post_Partum
    pm_post_partum_to_implant = [
        [
            [0.6, 0.5],
            [0.4, 0.3],
            [0.2, 0.1]
        ],
        [
            [0.1, 0.4],
            [0.2, 0.5],
            [0.3, 0.6]
        ],
    ]
    post_partum_to_implant = Choice(Broadcast_Event=USE_IMPLANT,
                                Years=[1995, 2005],
                                Ages=[15, 25, 35],
                                Parity=[0, 5],
                                ProbabilityMatrix=pm_post_partum_to_implant)

    pm_post_partum_to_pill = [
        [
            [0.0, 0.1, 0.2],
            [0.7, 0.8, 0.9]
        ],
        [
            [0.8, 0.6, 0.4],
            [0.2, 0.4, 0.8]
        ]
    ]
    post_partum_to_pill = Choice(Broadcast_Event=USE_PILL,
                            Years=[1995, 2020],
                            Ages=[20, 40],
                            Parity=[0, 2, 6],
                            ProbabilityMatrix=pm_post_partum_to_pill)

    pm_post_partum_to_withdrawal = [
        [
            [0.1]
        ],
        [
            [0.9]
        ]
    ]
    post_partum_to_withdrawal = Choice(Broadcast_Event=USE_WITHDRAWAL,
                                  Years=[1990, 2030],
                                  Ages=[20],
                                  Parity=[4],
                                  ProbabilityMatrix=pm_post_partum_to_withdrawal)

    choices_post_partum = []
    choices_post_partum.append(post_partum_to_implant)
    choices_post_partum.append(post_partum_to_pill)
    choices_post_partum.append(post_partum_to_withdrawal)

    filters_post_partum = filters

    rcm_from_post_partum = RandomChoiceMatrix(Choices=choices_post_partum, Filters=filters_post_partum)

    # On Implant
    pm_implant_to_implant = [
        [
            [0.1, 0.5],
            [0.4, 0.2],
            [0.6, 0.1]
        ],
        [
            [0.1, 0.0],
            [0.1, 0.6],
            [0.3, 0.6]
        ],
    ]
    implant_to_implant = Choice(Broadcast_Event=USE_IMPLANT,
                                Years=[1995, 2005],
                                Ages=[15, 25, 35],
                                Parity=[0, 5],
                                ProbabilityMatrix=pm_implant_to_implant)

    pm_implant_to_pill = [
        [
            [0.1]
        ],
        [
            [0.9]
        ]
    ]
    implant_to_pill = Choice(Broadcast_Event=USE_PILL,
                            Years=[1995, 2020],
                            Ages=[20],
                            Parity=[0],
                            ProbabilityMatrix=pm_implant_to_pill)

    pm_implant_to_withdrawal = [
        [
            [0.0, 0.1, 0.2],
            [0.7, 0.8, 0.9]
        ],
        [
            [0.8, 0.6, 0.4],
            [0.2, 0.4, 0.8]
        ]
    ]
    implant_to_withdrawal = Choice(Broadcast_Event=USE_WITHDRAWAL,
                                  Years=[1990, 2030],
                                  Ages=[20, 50],
                                  Parity=[0, 2, 4],
                                  ProbabilityMatrix=pm_implant_to_withdrawal)

    choices_implant = []
    choices_implant.append(implant_to_implant)
    choices_implant.append(implant_to_pill)
    choices_implant.append(implant_to_withdrawal)

    filters_implant = filters

    rcm_from_implant = RandomChoiceMatrix(Choices=choices_implant, Filters=filters_implant)

    # On Pill
    pm_pill_to_implant = [
        [
            [0.0, 0.4],
            [0.4, 0.2],
            [0.3, 0.1]
        ],
        [
            [0.1, 0.0],
            [0.9, 0.6],
            [0.3, 0.2]
        ],
    ]
    pill_to_implant = Choice(Broadcast_Event=USE_IMPLANT,
                                Years=[1990, 2053],
                                Ages=[15, 35, 55],
                                Parity=[0, 3],
                                ProbabilityMatrix=pm_pill_to_implant)

    pm_pill_to_pill = [
        [
            [0.8]
        ],
        [
            [0.5]
        ]
    ]
    pill_to_pill = Choice(Broadcast_Event=USE_PILL,
                            Years=[1990, 2020],
                            Ages=[20],
                            Parity=[0],
                            ProbabilityMatrix=pm_pill_to_pill)

    pm_pill_to_withdrawal = [
        [
            [0.01, 0.1, 0.2],
            [0.7, 0.2, 0.9]
        ],
        [
            [0.8, 0.9, 0.4],
            [0.2, 0.3, 0.8]
        ]
    ]
    pill_to_withdrawal = Choice(Broadcast_Event=USE_WITHDRAWAL,
                                  Years=[1990, 2030],
                                  Ages=[20, 45],
                                  Parity=[0, 1, 2],
                                  ProbabilityMatrix=pm_pill_to_withdrawal)

    choices_pill = []
    choices_pill.append(pill_to_implant)
    choices_pill.append(pill_to_pill)
    choices_pill.append(pill_to_withdrawal)

    filters_pill = filters

    rcm_from_pill = RandomChoiceMatrix(Choices=choices_pill, Filters=filters_pill)

    # On Withdrawal
    pm_withdrawal_to_implant = [
        [
            [0.0, 0.4],
            [0.4, 0.6],
            [0.9, 0.2]
        ],
        [
            [0.1, 1.0],
            [0.6, 0.6],
            [0.3, 0.2]
        ],
    ]
    withdrawal_to_implant = Choice(Broadcast_Event=USE_IMPLANT,
                                Years=[1990, 2053],
                                Ages=[15, 35, 55],
                                Parity=[0, 3],
                                ProbabilityMatrix=pm_withdrawal_to_implant)

    pm_withdrawal_to_pill = [
        [
            [0.8]
        ],
        [
            [0.6]
        ]
    ]
    withdrawal_to_pill = Choice(Broadcast_Event=USE_PILL,
                            Years=[1990, 2020],
                            Ages=[20],
                            Parity=[0],
                            ProbabilityMatrix=pm_withdrawal_to_pill)

    pm_withdrawal_to_withdrawal = [
        [
            [0.0, 0.1, 0.2],
            [0.7, 0.2, 0.9]
        ],
        [
            [0.1, 0.9, 0.1],
            [0.2, 0.5, 0.8]
        ]
    ]
    withdrawal_to_withdrawal = Choice(Broadcast_Event=USE_WITHDRAWAL,
                                  Years=[1990, 2030],
                                  Ages=[20, 45],
                                  Parity=[0, 1, 2],
                                  ProbabilityMatrix=pm_withdrawal_to_withdrawal)

    choices_withdrawal = []
    choices_withdrawal.append(withdrawal_to_implant)
    choices_withdrawal.append(withdrawal_to_pill)
    choices_withdrawal.append(withdrawal_to_withdrawal)

    filters_withdrawal  = filters

    rcm_from_withdrawal = RandomChoiceMatrix(Choices=choices_withdrawal, Filters=filters_withdrawal)

    rcm_list = []

    # we need to load the matrix from a file with defined format and turn it into rcm_list automatically.
    rcm_list.append( ( Choose_Next_Method_Currently_Under_Age    , rcm_from_under_age   ) )
    rcm_list.append( ( Choose_Next_Method_Currently_Post_Partum  , rcm_from_post_partum ) )
    rcm_list.append( ( Choose_Next_Method_Currently_On_Implant   , rcm_from_implant    ) )
    rcm_list.append( ( Choose_Next_Method_Currently_On_Pill      , rcm_from_pill        ) )
    rcm_list.append( ( Choose_Next_Method_Currently_On_Withdrawal, rcm_from_withdrawal  ) )
    #rcm_list.append( ( Choose_Next_Method_Currently_On_None      , rcm_from_none        ) )

    return rcm_list


# -----------------------------------------------------------------------------
# --- MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating campaign file for a FP sim")

    con_list = CreateContraceptives()
    rc_list = CreateRandomChoiceMatrixList()

    c = GenerateCampaignFP( con_list, rc_list )
    #c.save_to_file("/work/dev/trunk1/Regression/FP/2_FP_SimpleCampaign/campaign")
    if not os.path.exists("output"):
        os.mkdir("output")
    # use_defaults=False will save everything including the default values in the campaign file which will make the
    # campaign file looks messy but it will not trigger issue:
    # https://github.com/InstituteforDiseaseModeling/DtkTrunk/issues/3811
    # c.save_to_file(os.path.join("output", "campaign-RCM"), use_defaults=False)
    c.save_to_file(os.path.join("json", "campaign"))

    print("Done")

