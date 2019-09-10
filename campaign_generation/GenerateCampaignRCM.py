#!/usr/bin/python

import os
from CampaignClass import *
from CampaignEnum import *
#from InterventionClass import *

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
IP_NORPLANT    = "CurrentStatus:NORPLANT"
IP_PILL        = "CurrentStatus:PILL"
IP_WITHDRAWAL  = "CurrentStatus:WITHDRAWAL"
IP_NONE        = "CurrentStatus:NONE"
        
Should_Not_Be_Broadcasted = "Should_Not_Be_Broadcasted"
Choose_Next_Method_Currently_Under_Age     = "Choose_Next_Method_Currently_Under_Age"
Choose_Next_Method_Currently_Post_Partum   = "Choose_Next_Method_Currently_Post_Partum"
Choose_Next_Method_Currently_On_Norplant   = "Choose_Next_Method_Currently_On_Norplant"
Choose_Next_Method_Currently_On_Pill       = "Choose_Next_Method_Currently_On_Pill"
Choose_Next_Method_Currently_On_Withdrawal = "Choose_Next_Method_Currently_On_Withdrawal"
Choose_Next_Method_Currently_On_None       = "Choose_Next_Method_Currently_On_None"

USE_NORPLANT   = "Use_Norplant"
USE_PILL       = "Use_Pill"
USE_WITHDRAWAL = "Use_Withdrawal"
USE_NONE       = "Use_None"


# -----------------------------------------------------------------------------
# --- POPULATION INITIALIZATION - Functions for initializing the population
# ---                             with contraceptives
# -----------------------------------------------------------------------------

# Initialize women under 15 at the start of the simulation with a delay so that
# they choose a contraceptive about the time they turn 15.
# !!! COULD PROBABLY USE REFINEMENT !!!

def InitializeUnderAgeWomen( campaign ):
    be_cnm_under_age = BroadcastEvent(Choose_Next_Method_Currently_Under_Age)
    
    mean_delay_list = [ 13.0*365.0, 8.0*365.0, 3.0*365.0 ]
    age = 0.0
    
    for mean in mean_delay_list:
        delay = DelayedIntervention( Delay_Period_Distribution = DelayedIntervention_Delay_Period_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                     Delay_Period_Gaussian_Mean = mean,
                                     Delay_Period_Gaussian_Std_Dev = 180.0,
                                     New_Property_Value = IP_UNDER_AGE,
                                     Actual_IndividualIntervention_Configs = [be_cnm_under_age] )
                                      
        ce = distributeIntervention( Target_Demographic = "ExplicitAgeRangesAndGender",
                                     Target_Gender = "Female",
                                     Target_Age_Min = age,
                                     Target_Age_Max = age+5.0,
                                     Intervention_Config = delay )
        campaign.add_campaign_event( ce )
        age = age + 5.0


# Initialize the possible mothers at the start of the simulation with a contraceptive.
# The quick solution is to have them choose a method as if they just turned 15.
# !!! COULD PROBABLY USE REFINEMENT !!!

def InitializePossibleMothers( campaign ):
    ce = distributeIntervention( Target_Demographic = "ExplicitAgeRangesAndGender",
                                 Target_Gender = "Female",
                                 Target_Age_Min = 15.0,
                                 Target_Age_Max = 50.0,
                                 Intervention_Config = BroadcastEvent(Choose_Next_Method_Currently_Under_Age) )
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

# ??? 2) DO WE WANT TO USE CONTRACEPTIVE INSTEAD OF DELAYED INTERVENTION?

def DistributeDelayToNewBorns( campaign ):
    be_cnm_under_age = BroadcastEvent(Choose_Next_Method_Currently_Under_Age)
    
    delay = DelayedIntervention( Delay_Period_Distribution = DelayedIntervention_Delay_Period_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                 Delay_Period_Gaussian_Mean = 15.0*365.0,
                                 Delay_Period_Gaussian_Std_Dev = 180.0,
                                 New_Property_Value = IP_UNDER_AGE,
                                 Actual_IndividualIntervention_Configs = [be_cnm_under_age] )
                                  
    ce = distributeInterventionOnEvent( Target_Demographic = "ExplicitGender",
                                        Target_Gender = "Female",
                                        Trigger_Condition_List=["Births"],
                                        Intervention_Config = delay )
    campaign.add_campaign_event( ce )


# Distribute a "pregnant" contraceptive.  We distribute a contraceptive mainly
# to cause an individual's existing contraceptive to abort.  That is, this
# contraceptive will replace an existing one when women becomes pregnant.

# ??? WE DON'T NEED TO USE 'CONTRACEPTIVE'.  WE COULD USE BroadcastEvent.
# ??? WE JUST NEED SOMETHING TO CHANGE THE CurrentStatus to PREGNANT.

def DistributeContraceptiveToPregnantMothers( campaign ):

    con_pregnant = Contraceptive( Intervention_Name="Pregnant",
                                  New_Property_Value=IP_PREGNANT,
                                  Disqualifying_Properties=[ IP_POST_PARTUM ],
                                  Waning_Config=WaningEffectConstant(InitialEfficacy=1.0),
                                  Usage_Duration_Distribution=Contraceptive_Usage_Duration_Distribution_Enum.CONSTANT_DISTRIBUTION,
                                  Usage_Duration_Period = 10000,
                                  Usage_Expiration_Event = Should_Not_Be_Broadcasted )
                                  
    ce = distributeInterventionOnEvent( Target_Demographic = "ExplicitGender",
                                        Target_Gender = "Female",
                                        Trigger_Condition_List=["Pregnant"],
                                        Intervention_Config = con_pregnant )
    campaign.add_campaign_event( ce )

    
# Distribute a "post partunm" contraceptive to women when they give birth.

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
    DistributeDelayToNewBorns( campaign )

    DistributeInterventionList( campaign, randomChoiceList )
    DistributeInterventionList( campaign, contraceptiveList )
    
    DistributeContraceptiveToPregnantMothers( campaign )
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
    
    con_norplant   = Contraceptive( Intervention_Name="Norplant",
                                    New_Property_Value=IP_NORPLANT,
                                    Disqualifying_Properties=[ IP_PREGNANT ],
                                    Waning_Config=WaningEffectExponential(Initial_Effect=1.0,Decay_Time_Constant=365),
                                    Usage_Duration_Distribution = Contraceptive_Usage_Duration_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                    Usage_Duration_Gaussian_Mean = 180,
                                    Usage_Duration_Gaussian_Std_Dev = 30,
                                    Usage_Expiration_Event = Choose_Next_Method_Currently_On_Norplant )

    con_pill       = Contraceptive( Intervention_Name="Pill",
                                    New_Property_Value=IP_PILL,
                                    Disqualifying_Properties=[ IP_PREGNANT ],
                                    Waning_Config=WaningEffectBoxExponential(Initial_Effect=0.9,Box_Duration=90,Decay_Time_Constant=365),
                                    Usage_Duration_Distribution = Contraceptive_Usage_Duration_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                    Usage_Duration_Gaussian_Mean = 90,
                                    Usage_Duration_Gaussian_Std_Dev = 10,
                                    Usage_Expiration_Event = Choose_Next_Method_Currently_On_Pill )

    con_withdrawal = Contraceptive( Intervention_Name="Withdrawal",
                                    New_Property_Value=IP_WITHDRAWAL,
                                    Disqualifying_Properties=[ IP_PREGNANT ],
                                    Waning_Config=WaningEffectConstant(Initial_Effect=0.2),
                                    Usage_Duration_Distribution = Contraceptive_Usage_Duration_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                    Usage_Duration_Gaussian_Mean = 180,
                                    Usage_Duration_Gaussian_Std_Dev = 10,
                                    Usage_Expiration_Event = Choose_Next_Method_Currently_On_Withdrawal )
                                  
    con_none       = Contraceptive( Intervention_Name="None",
                                    New_Property_Value=IP_NONE,
                                    Disqualifying_Properties=[ IP_PREGNANT ],
                                    Waning_Config=WaningEffectConstant(Initial_Effect=0.0),
                                    Usage_Duration_Distribution = Contraceptive_Usage_Duration_Distribution_Enum.GAUSSIAN_DISTRIBUTION,
                                    Usage_Duration_Gaussian_Mean = 365,
                                    Usage_Duration_Gaussian_Std_Dev = 3,
                                    Usage_Expiration_Event = Choose_Next_Method_Currently_On_None )
                                    
    con_list = []
    con_list.append( (USE_NORPLANT  , con_norplant  ) )
    con_list.append( (USE_PILL      , con_pill      ) )
    con_list.append( (USE_WITHDRAWAL, con_withdrawal) )
    con_list.append( (USE_NONE      , con_none      ) )
    
    return con_list


# -----------------------------------------------------------------------------
# --- RANDOM CHOICES
# -----------------------------------------------------------------------------

# !!! I'm hardcoding this here for initial proof of concept.  In the future,
# !!! I expect the initial parameters for the RandomChoices to come from an
# !!! input file and then modified by calibration logic.  This will all happen
# !!! outside the campaign generation logic.

# NOTE: This is using the RandomChoice object but the logic that is using this
#       list will not need to change when we change it to RandomChoiceMatrix.

def CreateRandomChoiceList():
    rc_from_under_age   = RandomChoice( Choice_Names        =[ USE_NORPLANT, USE_PILL, USE_WITHDRAWAL, USE_NONE ],
                                        Choice_Probabilities=[          0.1,      0.2,            0.2,      0.5 ] )
    rc_from_post_partum = RandomChoice( Choice_Names        =[ USE_NORPLANT, USE_PILL, USE_WITHDRAWAL, USE_NONE ],
                                        Choice_Probabilities=[          0.2,      0.2,            0.1,      0.5 ] )
    rc_from_norplant    = RandomChoice( Choice_Names        =[ USE_NORPLANT, USE_PILL, USE_WITHDRAWAL, USE_NONE ],
                                        Choice_Probabilities=[          0.5,      0.3,            0.1,      0.1 ] )
    rc_from_pill        = RandomChoice( Choice_Names        =[ USE_NORPLANT, USE_PILL, USE_WITHDRAWAL, USE_NONE ],
                                        Choice_Probabilities=[          0.1,      0.5,            0.2,      0.2 ] )
    rc_from_withdrawal  = RandomChoice( Choice_Names        =[ USE_NORPLANT, USE_PILL, USE_WITHDRAWAL, USE_NONE ],
                                        Choice_Probabilities=[          0.1,      0.1,            0.4,      0.4 ] )
    rc_from_none        = RandomChoice( Choice_Names        =[ USE_NORPLANT, USE_PILL, USE_WITHDRAWAL, USE_NONE ],
                                        Choice_Probabilities=[          0.1,      0.2,            0.2,      0.5 ] )
                                        
    rc_list = []
    rc_list.append( ( Choose_Next_Method_Currently_Under_Age    , rc_from_under_age   ) )
    rc_list.append( ( Choose_Next_Method_Currently_Post_Partum  , rc_from_post_partum ) )
    rc_list.append( ( Choose_Next_Method_Currently_On_Norplant  , rc_from_norplant    ) )
    rc_list.append( ( Choose_Next_Method_Currently_On_Pill      , rc_from_pill        ) )
    rc_list.append( ( Choose_Next_Method_Currently_On_Withdrawal, rc_from_withdrawal  ) )
    rc_list.append( ( Choose_Next_Method_Currently_On_None      , rc_from_none        ) )
    
    return rc_list

# -----------------------------------------------------------------------------
# --- RANDOM CHOICES
# -----------------------------------------------------------------------------


def CreateRandomChoiceMatrixList():
    pm_norplant = [
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
    under_age_norplant = Choice( Broadcast_Event=USE_NORPLANT,
                                 Years=[ 1990, 2000 ],
                                 Ages=[ 15, 25, 40 ],
                                 Parity=[ 0, 3 ],
                                 ProbabilityMatrix=pm_norplant )
    
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
    choices_under_age.append( under_age_norplant )
    choices_under_age.append( under_age_pill )
    choices_under_age.append( under_age_withdrawal )
    
    filters_under_age = []
    
    rcm_from_under_age = RandomChoiceMatrix( Choices=choices_under_age, Filters=filters_under_age )

                                        
    rcm_list = []
    rcm_list.append( ( Choose_Next_Method_Currently_Under_Age    , rcm_from_under_age   ) )
    #rcm_list.append( ( Choose_Next_Method_Currently_Post_Partum  , rcm_from_post_partum ) )
    #rcm_list.append( ( Choose_Next_Method_Currently_On_Norplant  , rcm_from_norplant    ) )
    #rcm_list.append( ( Choose_Next_Method_Currently_On_Pill      , rcm_from_pill        ) )
    #rcm_list.append( ( Choose_Next_Method_Currently_On_Withdrawal, rcm_from_withdrawal  ) )
    #rcm_list.append( ( Choose_Next_Method_Currently_On_None      , rcm_from_none        ) )
    
    return rcm_list

# -----------------------------------------------------------------------------
# --- MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating campaign file for a FP sim")
    
    con_list = CreateContraceptives()
    #rc_list = CreateRandomChoiceList()
    rc_list = CreateRandomChoiceMatrixList()
    
    c = GenerateCampaignFP( con_list, rc_list )
    #c.save_to_file("/work/dev/trunk1/Regression/FP/2_FP_SimpleCampaign/campaign")
    c.save_to_file(os.path.join("json", "campaign"))
    
    print("Done")
    