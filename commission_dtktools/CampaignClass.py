from dtk.utils.Campaign.ClassValidator import ClassValidator
from dtk.utils.Campaign.utils.BaseCampaign import BaseCampaign
try:
    from CampaignEnum import *
except ImportError:
    from .CampaignEnum import *



class CampaignEvent(BaseCampaign):
    _definition = {
        'Event_Coordinator_Config': {
            'description': 'An object that specifies how the event is handled by the simulation. It specifies which Event Coordinator class will handle the event, and then configures the coordinator.',
            'type': 'idmAbstractType:EventCoordinator',
        },
        'Nodeset_Config': {
            'description': 'An object that specifies in which nodes the interventions will be distributed.',
            'type': 'idmAbstractType:NodeSet',
        },
        'Start_Day': {
            'default': 1,
            'description': "The day of the simulation to activate the event's event coordinator.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'class': 'CampaignEvent',
    }
    _validator = ClassValidator(_definition, 'CampaignEvent')

    def __init__(self, Event_Coordinator_Config=None, Nodeset_Config=None, Start_Day=1, **kwargs):
        super(CampaignEvent, self).__init__(**kwargs)
        self.Event_Coordinator_Config = Event_Coordinator_Config
        self.Nodeset_Config = Nodeset_Config
        self.Start_Day = Start_Day



class CampaignEventByYear(BaseCampaign):
    _definition = {
        'Event_Coordinator_Config': {
            'description': 'An object that specifies how the event is handled by the simulation. It specifies which Event Coordinator class will handle the event, and then configures the coordinator.',
            'type': 'idmAbstractType:EventCoordinator',
        },
        'Nodeset_Config': {
            'description': 'An object that specifies in which nodes the interventions will be distributed.',
            'type': 'idmAbstractType:NodeSet',
        },
        'Start_Year': {
            'default': 1800,
            'description': 'The year to start using the assortivity weighting matrix.',
            'max': 2200,
            'min': 1800,
            'type': 'float',
        },
        'class': 'CampaignEventByYear',
    }
    _validator = ClassValidator(_definition, 'CampaignEventByYear')

    def __init__(self, Event_Coordinator_Config=None, Nodeset_Config=None, Start_Year=1800, **kwargs):
        super(CampaignEventByYear, self).__init__(**kwargs)
        self.Event_Coordinator_Config = Event_Coordinator_Config
        self.Nodeset_Config = Nodeset_Config
        self.Start_Year = Start_Year



class BroadcastCoordinatorEvent(BaseCampaign):
    _definition = {
        'Broadcast_Event': {
            'default': '',
            'description': 'The name of the event to be broadcast. This event must be set in the **Custom_Coordinator_Events** configuration parameter. This cannot be assigned an empty string for the event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Coordinator_Events.*' or Built-in",
        },
        'Coordinator_Name': {
            'default': 'BroadcastCoordinatorEvent',
            'description': 'The unique identifying coordinator name, which is useful with the output report, ReportCoordinatorEventRecorder.csv.',
            'type': 'string',
        },
        'Cost_To_Consumer': {
            'default': 0,
            'description': 'The unit cost of broadcasting the event.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'class': 'BroadcastCoordinatorEvent',
    }
    _validator = ClassValidator(_definition, 'BroadcastCoordinatorEvent')

    def __init__(self, Broadcast_Event='', Coordinator_Name='BroadcastCoordinatorEvent', Cost_To_Consumer=0, **kwargs):
        super(BroadcastCoordinatorEvent, self).__init__(**kwargs)
        self.Broadcast_Event = Broadcast_Event
        self.Coordinator_Name = Coordinator_Name
        self.Cost_To_Consumer = Cost_To_Consumer



class CalendarEventCoordinator(BaseCampaign):
    _definition = {
        'Distribution_Coverages': {
            'ascending': 0,
            'default': [],
            'description': 'A vector of floats for the fraction of individuals that will receive this intervention in a CalendarEventCoordinator.',
            'max': 1,
            'min': 0,
            'type': 'Vector Float',
        },
        'Distribution_Times': {
            'ascending': 0,
            'default': [],
            'description': 'A vector of integers for simulation times at which distribution of events occurs in a CalendarEventCoordinator.',
            'max': 2147480000.0,
            'min': 1,
            'type': 'Vector Int',
        },
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'class': 'CalendarEventCoordinator',
    }
    _validator = ClassValidator(_definition, 'CalendarEventCoordinator')

    def __init__(self, Distribution_Coverages=[], Distribution_Times=[], Intervention_Config=None, Node_Property_Restrictions=[], Property_Restrictions=[], Property_Restrictions_Within_Node=[], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=CalendarEventCoordinator_Target_Demographic_Enum.Everyone, Target_Gender=CalendarEventCoordinator_Target_Gender_Enum.All, Target_Residents_Only=False, **kwargs):
        super(CalendarEventCoordinator, self).__init__(**kwargs)
        self.Distribution_Coverages = Distribution_Coverages
        self.Distribution_Times = Distribution_Times
        self.Intervention_Config = Intervention_Config
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only



class CommunityHealthWorkerEventCoordinator(BaseCampaign):
    _definition = {
        'Amount_In_Shipment': {
            'default': 2147480000.0,
            'description': 'The number of interventions (such as vaccine doses) that a health worker or clinic receives in a shipment.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'integer',
        },
        'Days_Between_Shipments': {
            'default': 3.40282e+38,
            'description': 'The number of days to wait before a clinic or health worker receives a new shipment of interventions (such as vaccine doses).',
            'max': 3.40282e+38,
            'min': 1,
            'type': 'float',
        },
        'Demographic_Coverage': {
            'default': 1,
            'description': 'The fraction of individuals in the target demographic that will receive this intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Duration': {
            'default': 3.40282e+38,
            'description': 'The number of days for an event coordinator to be active before it expires.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Initial_Amount_Constant': {
            'default': 6,
            'depends-on': {
                'Initial_Amount_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Initial_Amount_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'The distribution type to use for assigning the initial amount of interventions in stock.',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Initial_Amount_Exponential': {
            'default': 6,
            'depends-on': {
                'Initial_Amount_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Initial_Amount_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Initial_Amount_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Initial_Amount_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Initial_Amount_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Initial_Amount_Kappa': {
            'default': 1,
            'depends-on': {
                'Initial_Amount_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Initial_Amount_Lambda': {
            'default': 1,
            'depends-on': {
                'Initial_Amount_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Initial_Amount_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Initial_Amount_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Initial_Amount_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Initial_Amount_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Initial_Amount_Max': {
            'default': 1,
            'depends-on': {
                'Initial_Amount_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Initial_Amount_Mean_1': {
            'default': 1,
            'depends-on': {
                'Initial_Amount_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Initial_Amount_Mean_2': {
            'default': 1,
            'depends-on': {
                'Initial_Amount_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Initial_Amount_Min': {
            'default': 0,
            'depends-on': {
                'Initial_Amount_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Initial_Amount_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Initial_Amount_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Initial_Amount_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Initial_Amount_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Initial_Amount_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Initial_Amount_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Initial_Amount_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Initial_Amount_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'Max_Distributed_Per_Day': {
            'default': 2147480000.0,
            'description': 'The maximum number of interventions (such as vaccine doses) that can be distributed by health workers or clinics in a given day.',
            'max': 2147480000.0,
            'min': 1,
            'type': 'integer',
        },
        'Max_Stock': {
            'default': 2147480000.0,
            'description': 'The maximum number of interventions (such as vaccine doses) that can be stored by a health worker or clinic.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'integer',
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Trigger_Condition_List': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': [],
            'description': "The list of events that are of interest to the community health worker (CHW). If one of these events occurs, the individual or node is put into a queue to receive the CHW's intervention. The CHW processes the queue when the event coordinator is updated.",
            'type': 'Vector Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Waiting_Period': {
            'default': 3.40282e+38,
            'description': 'The number of days a person or node can be in the queue waiting to get the intervention from the community health worker (CHW).',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'class': 'CommunityHealthWorkerEventCoordinator',
    }
    _validator = ClassValidator(_definition, 'CommunityHealthWorkerEventCoordinator')

    def __init__(self, Amount_In_Shipment=2147480000.0, Days_Between_Shipments=3.40282e+38, Demographic_Coverage=1, Duration=3.40282e+38, Initial_Amount_Constant=6, Initial_Amount_Distribution=CommunityHealthWorkerEventCoordinator_Initial_Amount_Distribution_Enum.CONSTANT_DISTRIBUTION, Initial_Amount_Exponential=6, Initial_Amount_Gaussian_Mean=6, Initial_Amount_Gaussian_Std_Dev=1, Initial_Amount_Kappa=1, Initial_Amount_Lambda=1, Initial_Amount_Log_Normal_Mu=6, Initial_Amount_Log_Normal_Sigma=1, Initial_Amount_Max=1, Initial_Amount_Mean_1=1, Initial_Amount_Mean_2=1, Initial_Amount_Min=0, Initial_Amount_Peak_2_Value=1, Initial_Amount_Poisson_Mean=6, Initial_Amount_Proportion_0=1, Initial_Amount_Proportion_1=1, Intervention_Config=None, Max_Distributed_Per_Day=2147480000.0, Max_Stock=2147480000.0, Node_Property_Restrictions=[], Property_Restrictions=[], Property_Restrictions_Within_Node=[], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=CommunityHealthWorkerEventCoordinator_Target_Demographic_Enum.Everyone, Target_Gender=CommunityHealthWorkerEventCoordinator_Target_Gender_Enum.All, Target_Residents_Only=False, Trigger_Condition_List=[], Waiting_Period=3.40282e+38, **kwargs):
        super(CommunityHealthWorkerEventCoordinator, self).__init__(**kwargs)
        self.Amount_In_Shipment = Amount_In_Shipment
        self.Days_Between_Shipments = Days_Between_Shipments
        self.Demographic_Coverage = Demographic_Coverage
        self.Duration = Duration
        self.Initial_Amount_Constant = Initial_Amount_Constant
        self.Initial_Amount_Distribution = (Initial_Amount_Distribution.name if isinstance(Initial_Amount_Distribution, Enum) else Initial_Amount_Distribution)
        self.Initial_Amount_Exponential = Initial_Amount_Exponential
        self.Initial_Amount_Gaussian_Mean = Initial_Amount_Gaussian_Mean
        self.Initial_Amount_Gaussian_Std_Dev = Initial_Amount_Gaussian_Std_Dev
        self.Initial_Amount_Kappa = Initial_Amount_Kappa
        self.Initial_Amount_Lambda = Initial_Amount_Lambda
        self.Initial_Amount_Log_Normal_Mu = Initial_Amount_Log_Normal_Mu
        self.Initial_Amount_Log_Normal_Sigma = Initial_Amount_Log_Normal_Sigma
        self.Initial_Amount_Max = Initial_Amount_Max
        self.Initial_Amount_Mean_1 = Initial_Amount_Mean_1
        self.Initial_Amount_Mean_2 = Initial_Amount_Mean_2
        self.Initial_Amount_Min = Initial_Amount_Min
        self.Initial_Amount_Peak_2_Value = Initial_Amount_Peak_2_Value
        self.Initial_Amount_Poisson_Mean = Initial_Amount_Poisson_Mean
        self.Initial_Amount_Proportion_0 = Initial_Amount_Proportion_0
        self.Initial_Amount_Proportion_1 = Initial_Amount_Proportion_1
        self.Intervention_Config = Intervention_Config
        self.Max_Distributed_Per_Day = Max_Distributed_Per_Day
        self.Max_Stock = Max_Stock
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Trigger_Condition_List = Trigger_Condition_List
        self.Waiting_Period = Waiting_Period



class CoverageByNodeEventCoordinator(BaseCampaign):
    _definition = {
        'Coverage_By_Node': {
            'default': [],
            'description': 'An array of (nodeID, coverage) pairs configuring the demographic coverage of interventions by node for the targeted populations. The coverage value must be a float between 0 and 1.',
            'item_type': 'NodeIdAndCoverage',
            'type': 'Vector',
        },
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Number_Repetitions': {
            'default': 1,
            'description': 'The number of times an intervention is given, used with Timesteps_Between_Repetitions.',
            'max': 1000,
            'min': -1,
            'type': 'integer',
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Timesteps_Between_Repetitions': {
            'default': -1,
            'description': 'The repetition interval.',
            'max': 10000,
            'min': -1,
            'type': 'integer',
        },
        'class': 'CoverageByNodeEventCoordinator',
    }
    _validator = ClassValidator(_definition, 'CoverageByNodeEventCoordinator')

    def __init__(self, Coverage_By_Node=[], Intervention_Config=None, Node_Property_Restrictions=[], Number_Repetitions=1, Property_Restrictions=[], Property_Restrictions_Within_Node=[], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=CoverageByNodeEventCoordinator_Target_Demographic_Enum.Everyone, Target_Gender=CoverageByNodeEventCoordinator_Target_Gender_Enum.All, Target_Residents_Only=False, Timesteps_Between_Repetitions=-1, **kwargs):
        super(CoverageByNodeEventCoordinator, self).__init__(**kwargs)
        self.Coverage_By_Node = Coverage_By_Node
        self.Intervention_Config = Intervention_Config
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Number_Repetitions = Number_Repetitions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Timesteps_Between_Repetitions = Timesteps_Between_Repetitions



class DelayEventCoordinator(BaseCampaign):
    _definition = {
        'Coordinator_Name': {
            'default': 'DelayEventCoordinator',
            'description': 'The unique identifying coordinator name used to identify the different coordinators in reports.',
            'type': 'string',
        },
        'Delay_Complete_Event': {
            'default': '',
            'description': 'The delay completion event to be included in the ReportCoordinatorEventRecorder.csv output report, upon completion of the delay period.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Coordinator_Events.*' or Built-in",
        },
        'Delay_Period_Constant': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'The distribution type to use for assigning the delay period for distributing interventions.',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Delay_Period_Exponential': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Kappa': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Lambda': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Delay_Period_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Delay_Period_Max': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Mean_1': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Mean_2': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Min': {
            'default': 0,
            'depends-on': {
                'Delay_Period_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Duration': {
            'default': -1,
            'description': 'The time period (in days) that the triggered event coordinator is active before it expires.',
            'max': 3.40282e+38,
            'min': -1,
            'type': 'float',
        },
        'Sim_Types': ['*'],
        'Start_Trigger_Condition_List': {
            'default': [],
            'description': 'The trigger condition event list that when heard will start a new set of repetitions for the triggered event coordinator. The list cannot be empty.',
            'type': 'Vector Constrained String',
            'value_source': "'<configuration>:Custom_Coordinator_Events.*' or Built-in",
        },
        'Stop_Trigger_Condition_List': {
            'default': [],
            'description': 'The trigger condition event list that when heard will stop any repetitions for the triggered event coordinator until a start trigger condition event list is received. The list can be empty.',
            'type': 'Vector Constrained String',
            'value_source': "'<configuration>:Custom_Coordinator_Events.*' or Built-in",
        },
        'class': 'DelayEventCoordinator',
    }
    _validator = ClassValidator(_definition, 'DelayEventCoordinator')

    def __init__(self, Coordinator_Name='DelayEventCoordinator', Delay_Complete_Event='', Delay_Period_Constant=6, Delay_Period_Distribution=DelayEventCoordinator_Delay_Period_Distribution_Enum.CONSTANT_DISTRIBUTION, Delay_Period_Exponential=6, Delay_Period_Gaussian_Mean=6, Delay_Period_Gaussian_Std_Dev=1, Delay_Period_Kappa=1, Delay_Period_Lambda=1, Delay_Period_Log_Normal_Mu=6, Delay_Period_Log_Normal_Sigma=1, Delay_Period_Max=1, Delay_Period_Mean_1=1, Delay_Period_Mean_2=1, Delay_Period_Min=0, Delay_Period_Peak_2_Value=1, Delay_Period_Poisson_Mean=6, Delay_Period_Proportion_0=1, Delay_Period_Proportion_1=1, Duration=-1, Sim_Types=['*'], Start_Trigger_Condition_List=[], Stop_Trigger_Condition_List=[], **kwargs):
        super(DelayEventCoordinator, self).__init__(**kwargs)
        self.Coordinator_Name = Coordinator_Name
        self.Delay_Complete_Event = Delay_Complete_Event
        self.Delay_Period_Constant = Delay_Period_Constant
        self.Delay_Period_Distribution = (Delay_Period_Distribution.name if isinstance(Delay_Period_Distribution, Enum) else Delay_Period_Distribution)
        self.Delay_Period_Exponential = Delay_Period_Exponential
        self.Delay_Period_Gaussian_Mean = Delay_Period_Gaussian_Mean
        self.Delay_Period_Gaussian_Std_Dev = Delay_Period_Gaussian_Std_Dev
        self.Delay_Period_Kappa = Delay_Period_Kappa
        self.Delay_Period_Lambda = Delay_Period_Lambda
        self.Delay_Period_Log_Normal_Mu = Delay_Period_Log_Normal_Mu
        self.Delay_Period_Log_Normal_Sigma = Delay_Period_Log_Normal_Sigma
        self.Delay_Period_Max = Delay_Period_Max
        self.Delay_Period_Mean_1 = Delay_Period_Mean_1
        self.Delay_Period_Mean_2 = Delay_Period_Mean_2
        self.Delay_Period_Min = Delay_Period_Min
        self.Delay_Period_Peak_2_Value = Delay_Period_Peak_2_Value
        self.Delay_Period_Poisson_Mean = Delay_Period_Poisson_Mean
        self.Delay_Period_Proportion_0 = Delay_Period_Proportion_0
        self.Delay_Period_Proportion_1 = Delay_Period_Proportion_1
        self.Duration = Duration
        self.Sim_Types = Sim_Types
        self.Start_Trigger_Condition_List = Start_Trigger_Condition_List
        self.Stop_Trigger_Condition_List = Stop_Trigger_Condition_List



class GroupInterventionDistributionEventCoordinator(BaseCampaign):
    _definition = {
        'Demographic_Coverage': {
            'default': 1,
            'description': 'The fraction of individuals in the target demographic that will receive this intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Number_Repetitions': {
            'default': 1,
            'description': 'The number of times an intervention is given, used with Timesteps_Between_Repetitions.',
            'max': 1000,
            'min': -1,
            'type': 'integer',
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Disease_State': {
            'default': 'Everyone',
            'description': 'The disease state group targeted by this intervention.',
            'enum': ['Everyone', 'Infected', 'ActiveInfection', 'LatentInfection', 'MDR', 'TreatmentNaive', 'HasFailedTreatment', 'HIVNegative', 'ActiveHadTreatment'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Timesteps_Between_Repetitions': {
            'default': -1,
            'description': 'The repetition interval.',
            'max': 10000,
            'min': -1,
            'type': 'integer',
        },
        'class': 'GroupInterventionDistributionEventCoordinator',
    }
    _validator = ClassValidator(_definition, 'GroupInterventionDistributionEventCoordinator')

    def __init__(self, Demographic_Coverage=1, Intervention_Config=None, Node_Property_Restrictions=[], Number_Repetitions=1, Property_Restrictions=[], Property_Restrictions_Within_Node=[], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=GroupInterventionDistributionEventCoordinator_Target_Demographic_Enum.Everyone, Target_Disease_State=GroupInterventionDistributionEventCoordinator_Target_Disease_State_Enum.Everyone, Target_Gender=GroupInterventionDistributionEventCoordinator_Target_Gender_Enum.All, Target_Residents_Only=False, Timesteps_Between_Repetitions=-1, **kwargs):
        super(GroupInterventionDistributionEventCoordinator, self).__init__(**kwargs)
        self.Demographic_Coverage = Demographic_Coverage
        self.Intervention_Config = Intervention_Config
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Number_Repetitions = Number_Repetitions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Disease_State = (Target_Disease_State.name if isinstance(Target_Disease_State, Enum) else Target_Disease_State)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Timesteps_Between_Repetitions = Timesteps_Between_Repetitions



class GroupInterventionDistributionEventCoordinatorHIV(BaseCampaign):
    _definition = {
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Number_Repetitions': {
            'default': 1,
            'description': 'The number of times an intervention is given, used with Timesteps_Between_Repetitions.',
            'max': 1000,
            'min': -1,
            'type': 'integer',
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Time_Offset': {
            'default': 0,
            'description': 'This offset is used to determine the demographic coverage for HIVCoinfectionDistribution (used with GroupInterventionDistributionEventCoordinatorHIV), which is defined in the demographics file.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Timesteps_Between_Repetitions': {
            'default': -1,
            'description': 'The repetition interval.',
            'max': 10000,
            'min': -1,
            'type': 'integer',
        },
        'class': 'GroupInterventionDistributionEventCoordinatorHIV',
    }
    _validator = ClassValidator(_definition, 'GroupInterventionDistributionEventCoordinatorHIV')

    def __init__(self, Intervention_Config=None, Node_Property_Restrictions=[], Number_Repetitions=1, Property_Restrictions=[], Property_Restrictions_Within_Node=[], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=GroupInterventionDistributionEventCoordinatorHIV_Target_Demographic_Enum.Everyone, Target_Gender=GroupInterventionDistributionEventCoordinatorHIV_Target_Gender_Enum.All, Target_Residents_Only=False, Time_Offset=0, Timesteps_Between_Repetitions=-1, **kwargs):
        super(GroupInterventionDistributionEventCoordinatorHIV, self).__init__(**kwargs)
        self.Intervention_Config = Intervention_Config
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Number_Repetitions = Number_Repetitions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Time_Offset = Time_Offset
        self.Timesteps_Between_Repetitions = Timesteps_Between_Repetitions



class IncidenceEventCoordinator(BaseCampaign):
    _definition = {
        'Incidence_Counter': {
            'description': 'List of JSON objects for specifying the conditions and parameters that must be met for an incidence to be counted.',
            'type': 'object',
            'subclasses': 'IncidenceCounter',
        },
        'Number_Repetitions': {
            'default': 1,
            'description': 'The number of times an intervention is given, used with Timesteps_Between_Repetitions.',
            'max': 1000,
            'min': -1,
            'type': 'integer',
        },
        'Responder': {
            'description': 'List of JSON objects for specifying the threshold type, values, and the actions to take when the specified conditions and parameters have been met, as configured in the Incidence_Counter parameter.',
            'type': 'object',
            'subclasses': 'Responder',
        },
        'Timesteps_Between_Repetitions': {
            'default': -1,
            'description': 'The repetition interval.',
            'max': 10000,
            'min': -1,
            'type': 'integer',
        },
        'class': 'IncidenceEventCoordinator',
    }
    _validator = ClassValidator(_definition, 'IncidenceEventCoordinator')

    def __init__(self, Incidence_Counter=None, Number_Repetitions=1, Responder=None, Timesteps_Between_Repetitions=-1, **kwargs):
        super(IncidenceEventCoordinator, self).__init__(**kwargs)
        self.Incidence_Counter = Incidence_Counter
        self.Number_Repetitions = Number_Repetitions
        self.Responder = Responder
        self.Timesteps_Between_Repetitions = Timesteps_Between_Repetitions



class NChooserEventCoordinator(BaseCampaign):
    _definition = {
        'Distributions': {
            'default': [],
            'description': 'The ordered list of elements defining when, to whom, and how many interventions to distribute.',
            'item_type': 'TargetedDistribution',
            'type': 'Vector',
        },
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'class': 'NChooserEventCoordinator',
    }
    _validator = ClassValidator(_definition, 'NChooserEventCoordinator')

    def __init__(self, Distributions=[], Intervention_Config=None, **kwargs):
        super(NChooserEventCoordinator, self).__init__(**kwargs)
        self.Distributions = Distributions
        self.Intervention_Config = Intervention_Config



class NChooserEventCoordinatorHIV(BaseCampaign):
    _definition = {
        'Distributions': {
            'default': [],
            'description': 'The ordered list of elements defining when, to whom, and how many interventions to distribute.',
            'item_type': 'TargetedDistributionHIV',
            'type': 'Vector',
        },
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'class': 'NChooserEventCoordinatorHIV',
    }
    _validator = ClassValidator(_definition, 'NChooserEventCoordinatorHIV')

    def __init__(self, Distributions=[], Intervention_Config=None, **kwargs):
        super(NChooserEventCoordinatorHIV, self).__init__(**kwargs)
        self.Distributions = Distributions
        self.Intervention_Config = Intervention_Config



class NChooserEventCoordinatorSTI(BaseCampaign):
    _definition = {
        'Distributions': {
            'default': [],
            'description': 'The ordered list of elements defining when, to whom, and how many interventions to distribute.',
            'item_type': 'TargetedDistributionSTI',
            'type': 'Vector',
        },
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'class': 'NChooserEventCoordinatorSTI',
    }
    _validator = ClassValidator(_definition, 'NChooserEventCoordinatorSTI')

    def __init__(self, Distributions=[], Intervention_Config=None, **kwargs):
        super(NChooserEventCoordinatorSTI, self).__init__(**kwargs)
        self.Distributions = Distributions
        self.Intervention_Config = Intervention_Config



class ReferenceTrackingEventCoordinator(BaseCampaign):
    _definition = {
        'End_Year': {
            'default': 2200,
            'description': 'The final date at which this set of targeted coverages should be applied (expiration).',
            'max': 2200,
            'min': 1800,
            'type': 'float',
        },
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Time_Value_Map': {
            'description': 'Map of times (years) to coverages.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Update_Period': {
            'default': 365,
            'description': 'The time between distribution updates.',
            'max': 3650,
            'min': 1,
            'type': 'float',
        },
        'class': 'ReferenceTrackingEventCoordinator',
    }
    _validator = ClassValidator(_definition, 'ReferenceTrackingEventCoordinator')

    def __init__(self, End_Year=2200, Intervention_Config=None, Node_Property_Restrictions=[], Property_Restrictions=[], Property_Restrictions_Within_Node=[], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=ReferenceTrackingEventCoordinator_Target_Demographic_Enum.Everyone, Target_Gender=ReferenceTrackingEventCoordinator_Target_Gender_Enum.All, Target_Residents_Only=False, Time_Value_Map=None, Update_Period=365, **kwargs):
        super(ReferenceTrackingEventCoordinator, self).__init__(**kwargs)
        self.End_Year = End_Year
        self.Intervention_Config = Intervention_Config
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Time_Value_Map = Time_Value_Map
        self.Update_Period = Update_Period



class ReferenceTrackingEventCoordinatorHIV(BaseCampaign):
    _definition = {
        'End_Year': {
            'default': 2200,
            'description': 'The final date at which this set of targeted coverages should be applied (expiration).',
            'max': 2200,
            'min': 1800,
            'type': 'float',
        },
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Disease_State': {
            'default': 'Everyone',
            'description': 'An array of particular disease states used in the ReferenceTrackingEventCoordinatorHIV.',
            'enum': ['Everyone', 'HIV_Positive', 'HIV_Negative', 'Tested_Positive', 'Tested_Negative', 'Not_Tested_Or_Tested_Negative'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Time_Value_Map': {
            'description': 'Map of times (years) to coverages.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Update_Period': {
            'default': 365,
            'description': 'The time between distribution updates.',
            'max': 3650,
            'min': 1,
            'type': 'float',
        },
        'class': 'ReferenceTrackingEventCoordinatorHIV',
    }
    _validator = ClassValidator(_definition, 'ReferenceTrackingEventCoordinatorHIV')

    def __init__(self, End_Year=2200, Intervention_Config=None, Node_Property_Restrictions=[], Property_Restrictions=[], Property_Restrictions_Within_Node=[], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=ReferenceTrackingEventCoordinatorHIV_Target_Demographic_Enum.Everyone, Target_Disease_State=ReferenceTrackingEventCoordinatorHIV_Target_Disease_State_Enum.Everyone, Target_Gender=ReferenceTrackingEventCoordinatorHIV_Target_Gender_Enum.All, Target_Residents_Only=False, Time_Value_Map=None, Update_Period=365, **kwargs):
        super(ReferenceTrackingEventCoordinatorHIV, self).__init__(**kwargs)
        self.End_Year = End_Year
        self.Intervention_Config = Intervention_Config
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Disease_State = (Target_Disease_State.name if isinstance(Target_Disease_State, Enum) else Target_Disease_State)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Time_Value_Map = Time_Value_Map
        self.Update_Period = Update_Period



class StandardInterventionDistributionEventCoordinator(BaseCampaign):
    _definition = {
        'Demographic_Coverage': {
            'default': 1,
            'description': 'The fraction of individuals in the target demographic that will receive this intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Number_Repetitions': {
            'default': 1,
            'description': 'The number of times an intervention is given, used with Timesteps_Between_Repetitions.',
            'max': 1000,
            'min': -1,
            'type': 'integer',
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Timesteps_Between_Repetitions': {
            'default': -1,
            'description': 'The repetition interval.',
            'max': 10000,
            'min': -1,
            'type': 'integer',
        },
        'class': 'StandardInterventionDistributionEventCoordinator',
    }
    _validator = ClassValidator(_definition, 'StandardInterventionDistributionEventCoordinator')

    def __init__(self, Demographic_Coverage=1, Intervention_Config=None, Node_Property_Restrictions=[], Number_Repetitions=1, Property_Restrictions=[], Property_Restrictions_Within_Node=[], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=StandardInterventionDistributionEventCoordinator_Target_Demographic_Enum.Everyone, Target_Gender=StandardInterventionDistributionEventCoordinator_Target_Gender_Enum.All, Target_Residents_Only=False, Timesteps_Between_Repetitions=-1, **kwargs):
        super(StandardInterventionDistributionEventCoordinator, self).__init__(**kwargs)
        self.Demographic_Coverage = Demographic_Coverage
        self.Intervention_Config = Intervention_Config
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Number_Repetitions = Number_Repetitions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Timesteps_Between_Repetitions = Timesteps_Between_Repetitions



class SurveillanceEventCoordinator(BaseCampaign):
    _definition = {
        'Coordinator_Name': {
            'default': 'SurveillanceEventCoordinator',
            'description': 'The unique identifying coordinator name, which is useful with the output report, ReportSurveillanceEventRecorder.csv.',
            'type': 'string',
        },
        'Duration': {
            'default': -1,
            'description': "The number of days from when the surveillance event coordinator was created by the campaign event. Once the number of days has passed, the delay event coordinator will unregister for events and expire. The default value of '-1' = never expire.",
            'max': 3.40282e+38,
            'min': -1,
            'type': 'float',
        },
        'Incidence_Counter': {
            'description': 'List of JSON objects for specifying the conditions and parameters that must be met for an incidence to be counted.',
            'type': 'object',
            'subclasses': 'IncidenceCounterSurveillance',
        },
        'Responder': {
            'description': 'List of JSON objects for specifying the threshold type, values, and the actions to take when the specified conditions and parameters have been met, as configured in the Incidence_Counter parameter.',
            'type': 'object',
            'subclasses': 'ResponderSurveillance',
        },
        'Sim_Types': ['*'],
        'Start_Trigger_Condition_List': {
            'default': [],
            'description': 'The trigger event list, as specified in the **Custom_Coordinator_Events** config parameter, that will start **Incidence_Counter** counting events.',
            'type': 'Vector Constrained String',
            'value_source': "'<configuration>:Custom_Coordinator_Events.*' or Built-in",
        },
        'Stop_Trigger_Condition_List': {
            'default': [],
            'description': 'The broadcast event list, as specified in the **Custom_Coordinator_Events** config parameter, that will stop **Incidence_Counter** counting events.',
            'type': 'Vector Constrained String',
            'value_source': "'<configuration>:Custom_Coordinator_Events.*' or Built-in",
        },
        'class': 'SurveillanceEventCoordinator',
    }
    _validator = ClassValidator(_definition, 'SurveillanceEventCoordinator')

    def __init__(self, Coordinator_Name='SurveillanceEventCoordinator', Duration=-1, Incidence_Counter=None, Responder=None, Sim_Types=['*'], Start_Trigger_Condition_List=[], Stop_Trigger_Condition_List=[], **kwargs):
        super(SurveillanceEventCoordinator, self).__init__(**kwargs)
        self.Coordinator_Name = Coordinator_Name
        self.Duration = Duration
        self.Incidence_Counter = Incidence_Counter
        self.Responder = Responder
        self.Sim_Types = Sim_Types
        self.Start_Trigger_Condition_List = Start_Trigger_Condition_List
        self.Stop_Trigger_Condition_List = Stop_Trigger_Condition_List



class TriggeredEventCoordinator(BaseCampaign):
    _definition = {
        'Completion_Event': {
            'default': '',
            'description': 'The completion event list that will be broadcast every time the triggered event coordinator completes a set of repetitions.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Coordinator_Events.*' or Built-in",
        },
        'Coordinator_Name': {
            'default': 'TriggeredEventCoordinator',
            'description': 'The unique identifying coordinator name used to identify the different coordinators in reports.',
            'type': 'string',
        },
        'Duration': {
            'default': -1,
            'description': 'The time period (in days) that the triggered event coordinator is active before it expires.',
            'max': 3.40282e+38,
            'min': -1,
            'type': 'float',
        },
        'Intervention_Config': {
            'description': 'The nested JSON of the actual intervention to be distributed by this event coordinator.',
            'type': 'idmAbstractType:Intervention',
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Number_Repetitions': {
            'default': 1,
            'description': 'The number of times an intervention is given, used with Timesteps_Between_Repetitions.',
            'max': 1000,
            'min': -1,
            'type': 'integer',
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Sim_Types': ['*'],
        'Start_Trigger_Condition_List': {
            'default': [],
            'description': 'The trigger condition event list that when heard will start a new set of repetitions for the triggered event coordinator. The list cannot be empty.',
            'type': 'Vector Constrained String',
            'value_source': "'<configuration>:Custom_Coordinator_Events.*' or Built-in",
        },
        'Stop_Trigger_Condition_List': {
            'default': [],
            'description': 'The trigger condition event list that when heard will stop any repetitions for the triggered event coordinator until a start trigger condition event list is received. The list can be empty.',
            'type': 'Vector Constrained String',
            'value_source': "'<configuration>:Custom_Coordinator_Events.*' or Built-in",
        },
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Timesteps_Between_Repetitions': {
            'default': -1,
            'description': 'The repetition interval.',
            'max': 10000,
            'min': -1,
            'type': 'integer',
        },
        'class': 'TriggeredEventCoordinator',
    }
    _validator = ClassValidator(_definition, 'TriggeredEventCoordinator')

    def __init__(self, Completion_Event='', Coordinator_Name='TriggeredEventCoordinator', Duration=-1, Intervention_Config=None, Node_Property_Restrictions=[], Number_Repetitions=1, Property_Restrictions=[], Property_Restrictions_Within_Node=[], Sim_Types=['*'], Start_Trigger_Condition_List=[], Stop_Trigger_Condition_List=[], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=TriggeredEventCoordinator_Target_Demographic_Enum.Everyone, Target_Gender=TriggeredEventCoordinator_Target_Gender_Enum.All, Target_Residents_Only=False, Timesteps_Between_Repetitions=-1, **kwargs):
        super(TriggeredEventCoordinator, self).__init__(**kwargs)
        self.Completion_Event = Completion_Event
        self.Coordinator_Name = Coordinator_Name
        self.Duration = Duration
        self.Intervention_Config = Intervention_Config
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Number_Repetitions = Number_Repetitions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Sim_Types = Sim_Types
        self.Start_Trigger_Condition_List = Start_Trigger_Condition_List
        self.Stop_Trigger_Condition_List = Stop_Trigger_Condition_List
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Timesteps_Between_Repetitions = Timesteps_Between_Repetitions



class ARTBasic(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 1,
            'description': 'Unit cost per drug (unamortized).',
            'max': 99999,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Achieve_Viral_Suppression': {
            'default': 183,
            'description': 'The number of days after ART initiation over which infectiousness declines linearly until the ART_Viral_Suppression_Multiplier takes full effect.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'ARTBasic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['HIV_SIM', 'TBHIV_SIM'],
        'Viral_Suppression': {
            'default': 1,
            'description': 'If set to true (1), ART will suppress viral load and extend prognosis.',
            'type': 'bool',
        },
        'class': 'ARTBasic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'ARTBasic')

    def __init__(self, Cost_To_Consumer=1, Days_To_Achieve_Viral_Suppression=183, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='ARTBasic', New_Property_Value='', Sim_Types=['HIV_SIM', 'TBHIV_SIM'], Viral_Suppression=True, iv_type='IndividualTargeted', **kwargs):
        super(ARTBasic, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Achieve_Viral_Suppression = Days_To_Achieve_Viral_Suppression
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Viral_Suppression = Viral_Suppression
        self.iv_type = iv_type



class ARTDropout(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 1,
            'description': 'Unit cost per drug (unamortized).',
            'max': 99999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'ARTDropout',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['HIV_SIM', 'TBHIV_SIM'],
        'class': 'ARTDropout',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'ARTDropout')

    def __init__(self, Cost_To_Consumer=1, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='ARTDropout', New_Property_Value='', Sim_Types=['HIV_SIM', 'TBHIV_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(ARTDropout, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class ActiveDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'ActiveDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['TBHIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'ActiveDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'ActiveDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Event_Or_Config=ActiveDiagnostic_Event_Or_Config_Enum.Config, Intervention_Name='ActiveDiagnostic', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['TBHIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(ActiveDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class AdherentDrug(BaseCampaign):
    _definition = {
        'Adherence_Config': {
            'description': 'A list of nested JSON objects for the interventions to be distributed by this event coordinator.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per drug (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Dosing_Type': {
            'default': 'SingleDose',
            'description': 'The type of anti-malarial dosing to distribute in a drug intervention.',
            'enum': ['SingleDose', 'FullTreatmentCourse', 'Prophylaxis', 'SingleDoseWhenSymptom', 'FullTreatmentWhenSymptom', 'SingleDoseParasiteDetect', 'FullTreatmentParasiteDetect', 'SingleDoseNewDetectionTech', 'FullTreatmentNewDetectionTech'],
            'type': 'enum',
        },
        'Drug_Type': {
            'default': 'UNINITIALIZED STRING',
            'description': 'The type of drug to distribute in a drugs intervention.',
            'type': 'Constrained String',
            'value_source': '<configuration>:Malaria_Drug_Params.*',
        },
        'Intervention_Name': {
            'default': 'AdherentDrug',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Max_Dose_Consideration_Duration': {
            'default': 3.40282e+38,
            'description': 'The maximum number of days that an individual will consider taking the doses of the drug.',
            'max': 3.40282e+38,
            'min': 0.0416667,
            'type': 'float',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Non_Adherence_Distribution': {
            'ascending': 0,
            'default': [],
            'description': 'The non adherence probability value(s) assigned to the corresponding options in Non_Adherence_Options. The sum of non adherence distribution values must equal a total of 1.',
            'max': 1,
            'min': 0,
            'type': 'Vector Float',
        },
        'Non_Adherence_Options': {
            'default': 'NEXT_UPDATE',
            'description': 'Defines the action the person takes if they do not take a particular dose, are not adherent.',
            'enum': ['NEXT_UPDATE', 'NEXT_DOSAGE_TIME', 'LOST_TAKE_NEXT', 'STOP'],
            'type': 'Vector Enum',
        },
        'Sim_Types': ['MALARIA_SIM'],
        'Took_Dose_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The event that triggers the drug intervention campaign.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'class': 'AdherentDrug',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'AdherentDrug')

    def __init__(self, Adherence_Config=None, Cost_To_Consumer=10, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Dosing_Type=AdherentDrug_Dosing_Type_Enum.SingleDose, Drug_Type='UNINITIALIZED STRING', Intervention_Name='AdherentDrug', Max_Dose_Consideration_Duration=3.40282e+38, New_Property_Value='', Non_Adherence_Distribution=[], Non_Adherence_Options='NEXT_UPDATE', Sim_Types=['MALARIA_SIM'], Took_Dose_Event='', iv_type='IndividualTargeted', **kwargs):
        super(AdherentDrug, self).__init__(**kwargs)
        self.Adherence_Config = Adherence_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Dosing_Type = (Dosing_Type.name if isinstance(Dosing_Type, Enum) else Dosing_Type)
        self.Drug_Type = Drug_Type
        self.Intervention_Name = Intervention_Name
        self.Max_Dose_Consideration_Duration = Max_Dose_Consideration_Duration
        self.New_Property_Value = New_Property_Value
        self.Non_Adherence_Distribution = Non_Adherence_Distribution
        self.Non_Adherence_Options = Non_Adherence_Options
        self.Sim_Types = Sim_Types
        self.Took_Dose_Event = Took_Dose_Event
        self.iv_type = iv_type



class AgeDiagnostic(BaseCampaign):
    _definition = {
        'Age_Thresholds': {
            'default': [],
            'description': 'Used to associate age ranges for individuals.',
            'item_type': 'RangeThreshold',
            'type': 'Vector',
        },
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'AgeDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['HIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'AgeDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'AgeDiagnostic')

    def __init__(self, Age_Thresholds=[], Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Intervention_Name='AgeDiagnostic', New_Property_Value='', Sim_Types=['HIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(AgeDiagnostic, self).__init__(**kwargs)
        self.Age_Thresholds = Age_Thresholds
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class AntiTBDrug(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 1,
            'description': 'Unit cost per drug (unamortized).',
            'max': 99999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Dose_Interval': {
            'default': 1,
            'description': 'The interval between doses, in days.',
            'max': 99999,
            'min': 0,
            'type': 'float',
        },
        'Drug_CMax': {
            'default': 1,
            'description': 'The maximum drug concentration that can be used, and is in the same units as Drug_PKPD_C50.',
            'max': 10000,
            'min': 0,
            'type': 'float',
        },
        'Drug_PKPD_C50': {
            'default': 1,
            'description': 'The concentration at which drug killing rates are half of the maximum. Must use the same units as Drug_Cmax.',
            'max': 5000,
            'min': 0,
            'type': 'float',
        },
        'Drug_Type': {
            'default': 'DOTS',
            'description': 'Specifies the name of the drug treatment, such as DOTS or EmpiricTreatment, which can be useful for reporting from the simulation.',
            'enum': ['DOTS', 'DOTSImproved', 'EmpiricTreatment', 'FirstLineCombo', 'SecondLineCombo', 'ThirdLineCombo', 'LatentTreatment'],
            'type': 'enum',
        },
        'Drug_Vd': {
            'default': 1,
            'description': 'The volume of drug distribution. This value is the ratio of the volume of the second compartment to the volume of the first compartment in a two-compartment model, and is dimensionless.',
            'max': 10000,
            'min': 0,
            'type': 'float',
        },
        'Durability_Profile': {
            'default': 'FIXED_DURATION_CONSTANT_EFFECT',
            'description': 'The profile of durability decay.',
            'enum': ['FIXED_DURATION_CONSTANT_EFFECT', 'CONCENTRATION_VERSUS_TIME'],
            'type': 'enum',
        },
        'Fraction_Defaulters': {
            'default': 0,
            'description': 'The fraction of individuals who will not finish their drug treatment.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'AntiTBDrug',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Primary_Decay_Time_Constant': {
            'default': 1,
            'description': 'The primary decay time constant (in days) of the decay profile.',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Reduced_Transmit': {
            'default': 1,
            'description': 'The transmission reduction ratio, or the reduced infectiousness when an individual receives drugs.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Remaining_Doses': {
            'default': 0,
            'description': 'The remaining doses in an intervention; enter a negative number for unlimited doses.',
            'max': 999999,
            'min': -1,
            'type': 'integer',
        },
        'Secondary_Decay_Time_Constant': {
            'default': 1,
            'description': 'The secondary decay time constant of the durability profile.',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['TBHIV_SIM'],
        'TB_Drug_Cure_Rate': {
            'default': 0,
            'description': 'The daily probability of TB cure in an individual with drug-sensitive TB under drug treatment.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'TB_Drug_Inactivation_Rate': {
            'default': 0,
            'description': 'The daily rate at which treatment with an anti-TB drug causes inactivation in an individual with drug-sensitive TB.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'TB_Drug_Mortality_Rate': {
            'default': 0,
            'description': 'The daily rate at which treatment with an anti-TB drug causes death  in an individual with drug-sensitive TB.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'TB_Drug_Relapse_Rate': {
            'default': 0,
            'description': 'The daily probability of TB inactivation and subsequent relapse in an individual with drug-sensitive TB under drug treatment.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'TB_Drug_Resistance_Rate': {
            'default': 0,
            'description': 'The daily probability that an individual with drug-sensitive TB will acquire MDR-TB under drug treatment. Only individuals who return to the latent state or fail can acquire MDR-TB.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'AntiTBDrug',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'AntiTBDrug')

    def __init__(self, Cost_To_Consumer=1, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Dose_Interval=1, Drug_CMax=1, Drug_PKPD_C50=1, Drug_Type=AntiTBDrug_Drug_Type_Enum.DOTS, Drug_Vd=1, Durability_Profile=AntiTBDrug_Durability_Profile_Enum.FIXED_DURATION_CONSTANT_EFFECT, Fraction_Defaulters=0, Intervention_Name='AntiTBDrug', New_Property_Value='', Primary_Decay_Time_Constant=1, Reduced_Transmit=1, Remaining_Doses=0, Secondary_Decay_Time_Constant=1, Sim_Types=['TBHIV_SIM'], TB_Drug_Cure_Rate=0, TB_Drug_Inactivation_Rate=0, TB_Drug_Mortality_Rate=0, TB_Drug_Relapse_Rate=0, TB_Drug_Resistance_Rate=0, iv_type='IndividualTargeted', **kwargs):
        super(AntiTBDrug, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Dose_Interval = Dose_Interval
        self.Drug_CMax = Drug_CMax
        self.Drug_PKPD_C50 = Drug_PKPD_C50
        self.Drug_Type = (Drug_Type.name if isinstance(Drug_Type, Enum) else Drug_Type)
        self.Drug_Vd = Drug_Vd
        self.Durability_Profile = (Durability_Profile.name if isinstance(Durability_Profile, Enum) else Durability_Profile)
        self.Fraction_Defaulters = Fraction_Defaulters
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Primary_Decay_Time_Constant = Primary_Decay_Time_Constant
        self.Reduced_Transmit = Reduced_Transmit
        self.Remaining_Doses = Remaining_Doses
        self.Secondary_Decay_Time_Constant = Secondary_Decay_Time_Constant
        self.Sim_Types = Sim_Types
        self.TB_Drug_Cure_Rate = TB_Drug_Cure_Rate
        self.TB_Drug_Inactivation_Rate = TB_Drug_Inactivation_Rate
        self.TB_Drug_Mortality_Rate = TB_Drug_Mortality_Rate
        self.TB_Drug_Relapse_Rate = TB_Drug_Relapse_Rate
        self.TB_Drug_Resistance_Rate = TB_Drug_Resistance_Rate
        self.iv_type = iv_type



class AntimalarialDrug(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per drug (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Dosing_Type': {
            'default': 'SingleDose',
            'description': 'The type of anti-malarial dosing to distribute in a drug intervention.',
            'enum': ['SingleDose', 'FullTreatmentCourse', 'Prophylaxis', 'SingleDoseWhenSymptom', 'FullTreatmentWhenSymptom', 'SingleDoseParasiteDetect', 'FullTreatmentParasiteDetect', 'SingleDoseNewDetectionTech', 'FullTreatmentNewDetectionTech'],
            'type': 'enum',
        },
        'Drug_Type': {
            'default': 'UNINITIALIZED STRING',
            'description': 'The type of drug to distribute in a drugs intervention.',
            'type': 'Constrained String',
            'value_source': '<configuration>:Malaria_Drug_Params.*',
        },
        'Intervention_Name': {
            'default': 'AntimalarialDrug',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['MALARIA_SIM'],
        'class': 'AntimalarialDrug',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'AntimalarialDrug')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Dosing_Type=AntimalarialDrug_Dosing_Type_Enum.SingleDose, Drug_Type='UNINITIALIZED STRING', Intervention_Name='AntimalarialDrug', New_Property_Value='', Sim_Types=['MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(AntimalarialDrug, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Dosing_Type = (Dosing_Type.name if isinstance(Dosing_Type, Enum) else Dosing_Type)
        self.Drug_Type = Drug_Type
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class AntipoliovirusDrug(BaseCampaign):
    _definition = {
        'Adherence_Rate': {
            'default': 1,
            'description': 'Adherence rate for doses subsequent to first dose. Per-dose adherece rate for a dropout model.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': 'Unit cost per drug (unamortized).',
            'max': 99999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Dose_Interval': {
            'default': 1,
            'description': 'The interval between doses, in days.',
            'max': 99999,
            'min': 0,
            'type': 'float',
        },
        'Drug_CMax': {
            'default': 1,
            'description': 'The maximum drug concentration that can be used, and is in the same units as Drug_PKPD_C50.',
            'max': 10000,
            'min': 0,
            'type': 'float',
        },
        'Drug_PKPD_C50': {
            'default': 1,
            'description': 'The concentration at which drug killing rates are half of the maximum. Must use the same units as Drug_Cmax.',
            'max': 5000,
            'min': 0,
            'type': 'float',
        },
        'Drug_Vd': {
            'default': 1,
            'description': 'The volume of drug distribution. This value is the ratio of the volume of the second compartment to the volume of the first compartment in a two-compartment model, and is dimensionless.',
            'max': 10000,
            'min': 0,
            'type': 'float',
        },
        'Durability_Profile': {
            'default': 'FIXED_DURATION_CONSTANT_EFFECT',
            'description': 'The profile of durability decay.',
            'enum': ['FIXED_DURATION_CONSTANT_EFFECT', 'CONCENTRATION_VERSUS_TIME'],
            'type': 'enum',
        },
        'Fraction_Defaulters': {
            'default': 0,
            'description': 'The fraction of individuals who will not finish their drug treatment.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Infection_Duration_Efficacy': {
            'default': 1,
            'description': 'efficacy in reducing infection duration.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'AntipoliovirusDrug',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Primary_Decay_Time_Constant': {
            'default': 1,
            'description': 'The primary decay time constant (in days) of the decay profile.',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Remaining_Doses': {
            'default': 0,
            'description': 'The remaining doses in an intervention; enter a negative number for unlimited doses.',
            'max': 999999,
            'min': -1,
            'type': 'integer',
        },
        'Responder_Rate': {
            'default': 1,
            'description': 'Probability that an individual will have any response to the drug.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Secondary_Decay_Time_Constant': {
            'default': 1,
            'description': 'The secondary decay time constant of the durability profile.',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['POLIO_SIM'],
        'Titer_Efficacy': {
            'default': 1,
            'description': 'efficacy in reducing log10 tcid50 shed titer.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'AntipoliovirusDrug',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'AntipoliovirusDrug')

    def __init__(self, Adherence_Rate=1, Cost_To_Consumer=1, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Dose_Interval=1, Drug_CMax=1, Drug_PKPD_C50=1, Drug_Vd=1, Durability_Profile=AntipoliovirusDrug_Durability_Profile_Enum.FIXED_DURATION_CONSTANT_EFFECT, Fraction_Defaulters=0, Infection_Duration_Efficacy=1, Intervention_Name='AntipoliovirusDrug', New_Property_Value='', Primary_Decay_Time_Constant=1, Remaining_Doses=0, Responder_Rate=1, Secondary_Decay_Time_Constant=1, Sim_Types=['POLIO_SIM'], Titer_Efficacy=1, iv_type='IndividualTargeted', **kwargs):
        super(AntipoliovirusDrug, self).__init__(**kwargs)
        self.Adherence_Rate = Adherence_Rate
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Dose_Interval = Dose_Interval
        self.Drug_CMax = Drug_CMax
        self.Drug_PKPD_C50 = Drug_PKPD_C50
        self.Drug_Vd = Drug_Vd
        self.Durability_Profile = (Durability_Profile.name if isinstance(Durability_Profile, Enum) else Durability_Profile)
        self.Fraction_Defaulters = Fraction_Defaulters
        self.Infection_Duration_Efficacy = Infection_Duration_Efficacy
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Primary_Decay_Time_Constant = Primary_Decay_Time_Constant
        self.Remaining_Doses = Remaining_Doses
        self.Responder_Rate = Responder_Rate
        self.Secondary_Decay_Time_Constant = Secondary_Decay_Time_Constant
        self.Sim_Types = Sim_Types
        self.Titer_Efficacy = Titer_Efficacy
        self.iv_type = iv_type



class ArtificialDietHousingModification(BaseCampaign):
    _definition = {
        'Blocking_Config': {
            'description': 'The configuration of pre-feed mosquito repellency and waning for housing modification.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 8,
            'description': 'Unit cost per housing modification (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'ArtificialDietHousingModification',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of killing efficacy and waning for housing modification. Killing is conditional on NOT blocking the mosquito before feeding.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM'],
        'class': 'ArtificialDietHousingModification',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'ArtificialDietHousingModification')

    def __init__(self, Blocking_Config=None, Cost_To_Consumer=8, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='ArtificialDietHousingModification', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(ArtificialDietHousingModification, self).__init__(**kwargs)
        self.Blocking_Config = Blocking_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class BCGVaccine(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vaccine (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Efficacy_Is_Multiplicative': {
            'default': 1,
            'description': 'The overall vaccine efficacy when individuals receive more than one vaccine. When set to true (1), the vaccine efficacies are multiplied together; when set to false (0), the efficacies are additive.',
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'BCGVaccine',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['TBHIV_SIM'],
        'Vaccine_Take': {
            'default': 1,
            'description': 'The rate at which delivered vaccines will successfully stimulate an immune response and achieve the desired efficacy.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Vaccine_Take_Age_Decay_Rate': {
            'default': 1,
            'description': 'The exponential decline in vaccine take over time. The rate of decline is in units of 1/years.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Vaccine_Type': {
            'default': 'Generic',
            'description': 'The type of vaccine to distribute in a vaccine intervention.',
            'enum': ['Generic', 'TransmissionBlocking', 'AcquisitionBlocking', 'MortalityBlocking'],
            'type': 'enum',
        },
        'Waning_Config': {
            'description': 'The configuration of Ivermectin killing efficacy and waning over time.',
            'type': 'idmType:WaningEffect',
        },
        'class': 'BCGVaccine',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'BCGVaccine')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Efficacy_Is_Multiplicative=True, Intervention_Name='BCGVaccine', New_Property_Value='', Sim_Types=['TBHIV_SIM'], Vaccine_Take=1, Vaccine_Take_Age_Decay_Rate=1, Vaccine_Type=BCGVaccine_Vaccine_Type_Enum.Generic, Waning_Config=[], iv_type='IndividualTargeted', **kwargs):
        super(BCGVaccine, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Efficacy_Is_Multiplicative = Efficacy_Is_Multiplicative
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Vaccine_Take = Vaccine_Take
        self.Vaccine_Take_Age_Decay_Rate = Vaccine_Take_Age_Decay_Rate
        self.Vaccine_Type = (Vaccine_Type.name if isinstance(Vaccine_Type, Enum) else Vaccine_Type)
        self.Waning_Config = Waning_Config
        self.iv_type = iv_type



class BitingRisk(BaseCampaign):
    _definition = {
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'BitingRisk',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Risk_Constant': {
            'default': 6,
            'depends-on': {
                'Risk_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Risk_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'The distribution type to use for assigning the relative risk of being bitten by a mosquito to each individual.',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Risk_Exponential': {
            'default': 6,
            'depends-on': {
                'Risk_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Risk_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Risk_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Risk_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Risk_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Risk_Kappa': {
            'default': 1,
            'depends-on': {
                'Risk_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Risk_Lambda': {
            'default': 1,
            'depends-on': {
                'Risk_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Risk_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Risk_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Risk_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Risk_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Risk_Max': {
            'default': 1,
            'depends-on': {
                'Risk_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Risk_Mean_1': {
            'default': 1,
            'depends-on': {
                'Risk_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Risk_Mean_2': {
            'default': 1,
            'depends-on': {
                'Risk_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Risk_Min': {
            'default': 0,
            'depends-on': {
                'Risk_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Risk_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Risk_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Risk_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Risk_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Risk_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Risk_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Risk_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Risk_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM'],
        'class': 'BitingRisk',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'BitingRisk')

    def __init__(self, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='BitingRisk', New_Property_Value='', Risk_Constant=6, Risk_Distribution=BitingRisk_Risk_Distribution_Enum.CONSTANT_DISTRIBUTION, Risk_Exponential=6, Risk_Gaussian_Mean=6, Risk_Gaussian_Std_Dev=1, Risk_Kappa=1, Risk_Lambda=1, Risk_Log_Normal_Mu=6, Risk_Log_Normal_Sigma=1, Risk_Max=1, Risk_Mean_1=1, Risk_Mean_2=1, Risk_Min=0, Risk_Peak_2_Value=1, Risk_Poisson_Mean=6, Risk_Proportion_0=1, Risk_Proportion_1=1, Sim_Types=['VECTOR_SIM', 'MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(BitingRisk, self).__init__(**kwargs)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Risk_Constant = Risk_Constant
        self.Risk_Distribution = (Risk_Distribution.name if isinstance(Risk_Distribution, Enum) else Risk_Distribution)
        self.Risk_Exponential = Risk_Exponential
        self.Risk_Gaussian_Mean = Risk_Gaussian_Mean
        self.Risk_Gaussian_Std_Dev = Risk_Gaussian_Std_Dev
        self.Risk_Kappa = Risk_Kappa
        self.Risk_Lambda = Risk_Lambda
        self.Risk_Log_Normal_Mu = Risk_Log_Normal_Mu
        self.Risk_Log_Normal_Sigma = Risk_Log_Normal_Sigma
        self.Risk_Max = Risk_Max
        self.Risk_Mean_1 = Risk_Mean_1
        self.Risk_Mean_2 = Risk_Mean_2
        self.Risk_Min = Risk_Min
        self.Risk_Peak_2_Value = Risk_Peak_2_Value
        self.Risk_Poisson_Mean = Risk_Poisson_Mean
        self.Risk_Proportion_0 = Risk_Proportion_0
        self.Risk_Proportion_1 = Risk_Proportion_1
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class BroadcastEvent(BaseCampaign):
    _definition = {
        'Broadcast_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The event that should occur at the end of the delay period.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'BroadcastEvent',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['*'],
        'class': 'BroadcastEvent',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'BroadcastEvent')

    def __init__(self, Broadcast_Event='', Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='BroadcastEvent', New_Property_Value='', Sim_Types=['*'], iv_type='IndividualTargeted', **kwargs):
        super(BroadcastEvent, self).__init__(**kwargs)
        self.Broadcast_Event = Broadcast_Event
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class BroadcastEventToOtherNodes(BaseCampaign):
    _definition = {
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Event_Trigger': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The name of the event to broadcast to selected nodes.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Include_My_Node': {
            'default': 0,
            'description': 'Set to true (1) to broadcast the event to the current node.',
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'BroadcastEventToOtherNodes',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Max_Distance_To_Other_Nodes_Km': {
            'default': 3.40282e+38,
            'description': 'The maximum distance, in kilometers, to the destination node for the node to be selected.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Node_Selection_Type': {
            'default': 'DISTANCE_ONLY',
            'description': 'The method by which to select nodes to receive the event.',
            'enum': ['DISTANCE_ONLY', 'MIGRATION_NODES_ONLY', 'DISTANCE_AND_MIGRATION'],
            'type': 'enum',
        },
        'Sim_Types': ['*'],
        'class': 'BroadcastEventToOtherNodes',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'BroadcastEventToOtherNodes')

    def __init__(self, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Event_Trigger='', Include_My_Node=False, Intervention_Name='BroadcastEventToOtherNodes', Max_Distance_To_Other_Nodes_Km=3.40282e+38, New_Property_Value='', Node_Selection_Type=BroadcastEventToOtherNodes_Node_Selection_Type_Enum.DISTANCE_ONLY, Sim_Types=['*'], iv_type='IndividualTargeted', **kwargs):
        super(BroadcastEventToOtherNodes, self).__init__(**kwargs)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Event_Trigger = Event_Trigger
        self.Include_My_Node = Include_My_Node
        self.Intervention_Name = Intervention_Name
        self.Max_Distance_To_Other_Nodes_Km = Max_Distance_To_Other_Nodes_Km
        self.New_Property_Value = New_Property_Value
        self.Node_Selection_Type = (Node_Selection_Type.name if isinstance(Node_Selection_Type, Enum) else Node_Selection_Type)
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class CD4Diagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'CD4_Thresholds': {
            'default': [],
            'description': 'This parameter associates ranges of CD4 counts with events that should occur for individuals whose CD4 counts fall into those ranges.',
            'item_type': 'RangeThreshold',
            'type': 'Vector',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'CD4Diagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['HIV_SIM', 'TBHIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'CD4Diagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'CD4Diagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, CD4_Thresholds=[], Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Intervention_Name='CD4Diagnostic', New_Property_Value='', Sim_Types=['HIV_SIM', 'TBHIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(CD4Diagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.CD4_Thresholds = CD4_Thresholds
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class Contraceptive(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 0,
            'description': 'TBD',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'Contraceptive',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['*'],
        'Usage_Duration_Constant': {
            'default': 6,
            'depends-on': {
                'Usage_Duration_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Usage_Duration_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'TBD',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Usage_Duration_Exponential': {
            'default': 6,
            'depends-on': {
                'Usage_Duration_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Usage_Duration_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Usage_Duration_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Usage_Duration_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Usage_Duration_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Usage_Duration_Kappa': {
            'default': 1,
            'depends-on': {
                'Usage_Duration_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Usage_Duration_Lambda': {
            'default': 1,
            'depends-on': {
                'Usage_Duration_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Usage_Duration_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Usage_Duration_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Usage_Duration_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Usage_Duration_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Usage_Duration_Max': {
            'default': 1,
            'depends-on': {
                'Usage_Duration_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Usage_Duration_Mean_1': {
            'default': 1,
            'depends-on': {
                'Usage_Duration_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Usage_Duration_Mean_2': {
            'default': 1,
            'depends-on': {
                'Usage_Duration_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Usage_Duration_Min': {
            'default': 0,
            'depends-on': {
                'Usage_Duration_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Usage_Duration_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Usage_Duration_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Usage_Duration_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Usage_Duration_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Usage_Duration_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Usage_Duration_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Usage_Duration_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Usage_Duration_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Usage_Expiration_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'TBD',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Waning_Config': {
            'description': 'TBD',
            'type': 'idmType:WaningEffect',
        },
        'class': 'Contraceptive',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'Contraceptive')

    def __init__(self, Cost_To_Consumer=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='Contraceptive', New_Property_Value='', Sim_Types=['*'], Usage_Duration_Constant=6, Usage_Duration_Distribution=Contraceptive_Usage_Duration_Distribution_Enum.CONSTANT_DISTRIBUTION, Usage_Duration_Exponential=6, Usage_Duration_Gaussian_Mean=6, Usage_Duration_Gaussian_Std_Dev=1, Usage_Duration_Kappa=1, Usage_Duration_Lambda=1, Usage_Duration_Log_Normal_Mu=6, Usage_Duration_Log_Normal_Sigma=1, Usage_Duration_Max=1, Usage_Duration_Mean_1=1, Usage_Duration_Mean_2=1, Usage_Duration_Min=0, Usage_Duration_Peak_2_Value=1, Usage_Duration_Poisson_Mean=6, Usage_Duration_Proportion_0=1, Usage_Duration_Proportion_1=1, Usage_Expiration_Event='', Waning_Config=[], iv_type='IndividualTargeted', **kwargs):
        super(Contraceptive, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Usage_Duration_Constant = Usage_Duration_Constant
        self.Usage_Duration_Distribution = (Usage_Duration_Distribution.name if isinstance(Usage_Duration_Distribution, Enum) else Usage_Duration_Distribution)
        self.Usage_Duration_Exponential = Usage_Duration_Exponential
        self.Usage_Duration_Gaussian_Mean = Usage_Duration_Gaussian_Mean
        self.Usage_Duration_Gaussian_Std_Dev = Usage_Duration_Gaussian_Std_Dev
        self.Usage_Duration_Kappa = Usage_Duration_Kappa
        self.Usage_Duration_Lambda = Usage_Duration_Lambda
        self.Usage_Duration_Log_Normal_Mu = Usage_Duration_Log_Normal_Mu
        self.Usage_Duration_Log_Normal_Sigma = Usage_Duration_Log_Normal_Sigma
        self.Usage_Duration_Max = Usage_Duration_Max
        self.Usage_Duration_Mean_1 = Usage_Duration_Mean_1
        self.Usage_Duration_Mean_2 = Usage_Duration_Mean_2
        self.Usage_Duration_Min = Usage_Duration_Min
        self.Usage_Duration_Peak_2_Value = Usage_Duration_Peak_2_Value
        self.Usage_Duration_Poisson_Mean = Usage_Duration_Poisson_Mean
        self.Usage_Duration_Proportion_0 = Usage_Duration_Proportion_0
        self.Usage_Duration_Proportion_1 = Usage_Duration_Proportion_1
        self.Usage_Expiration_Event = Usage_Expiration_Event
        self.Waning_Config = Waning_Config
        self.iv_type = iv_type



class ControlledVaccine(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vaccine (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Distributed_Event_Trigger': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The name of the event to be broadcast when the intervention is distributed to an individual. See the list of available events for possible values.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Duration_To_Wait_Before_Revaccination': {
            'default': 3.40282e+38,
            'description': 'The length of time, in days, to wait before revaccinating an individual.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Efficacy_Is_Multiplicative': {
            'default': 1,
            'description': 'The overall vaccine efficacy when individuals receive more than one vaccine. When set to true (1), the vaccine efficacies are multiplied together; when set to false (0), the efficacies are additive.',
            'type': 'bool',
        },
        'Expired_Event_Trigger': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The name of the event to be broadcast when the intervention is distributed to an individual. See the list of available events for possible values.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Intervention_Name': {
            'default': 'ControlledVaccine',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['*'],
        'Vaccine_Take': {
            'default': 1,
            'description': 'The rate at which delivered vaccines will successfully stimulate an immune response and achieve the desired efficacy.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Vaccine_Type': {
            'default': 'Generic',
            'description': 'The type of vaccine to distribute in a vaccine intervention.',
            'enum': ['Generic', 'TransmissionBlocking', 'AcquisitionBlocking', 'MortalityBlocking'],
            'type': 'enum',
        },
        'Waning_Config': {
            'description': 'The configuration of Ivermectin killing efficacy and waning over time.',
            'type': 'idmType:WaningEffect',
        },
        'class': 'ControlledVaccine',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'ControlledVaccine')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Distributed_Event_Trigger='', Dont_Allow_Duplicates=False, Duration_To_Wait_Before_Revaccination=3.40282e+38, Efficacy_Is_Multiplicative=True, Expired_Event_Trigger='', Intervention_Name='ControlledVaccine', New_Property_Value='', Sim_Types=['*'], Vaccine_Take=1, Vaccine_Type=ControlledVaccine_Vaccine_Type_Enum.Generic, Waning_Config=[], iv_type='IndividualTargeted', **kwargs):
        super(ControlledVaccine, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Distributed_Event_Trigger = Distributed_Event_Trigger
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Duration_To_Wait_Before_Revaccination = Duration_To_Wait_Before_Revaccination
        self.Efficacy_Is_Multiplicative = Efficacy_Is_Multiplicative
        self.Expired_Event_Trigger = Expired_Event_Trigger
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Vaccine_Take = Vaccine_Take
        self.Vaccine_Type = (Vaccine_Type.name if isinstance(Vaccine_Type, Enum) else Vaccine_Type)
        self.Waning_Config = Waning_Config
        self.iv_type = iv_type



class DelayedIntervention(BaseCampaign):
    _definition = {
        'Actual_IndividualIntervention_Configs': {
            'description': 'An array of nested interventions to be distributed at the end of a delay period, to covered fraction of the population.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Coverage': {
            'default': 1,
            'description': 'The proportion of individuals who receive the DelayedIntervention that actually receive the configured interventions.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Constant': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'The distribution type to use for assigning the delay period for distributing interventions.',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Delay_Period_Exponential': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Kappa': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Lambda': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Delay_Period_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Delay_Period_Max': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Mean_1': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Mean_2': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Min': {
            'default': 0,
            'depends-on': {
                'Delay_Period_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'DelayedIntervention',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['*'],
        'class': 'DelayedIntervention',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'DelayedIntervention')

    def __init__(self, Actual_IndividualIntervention_Configs=None, Coverage=1, Delay_Period_Constant=6, Delay_Period_Distribution=DelayedIntervention_Delay_Period_Distribution_Enum.CONSTANT_DISTRIBUTION, Delay_Period_Exponential=6, Delay_Period_Gaussian_Mean=6, Delay_Period_Gaussian_Std_Dev=1, Delay_Period_Kappa=1, Delay_Period_Lambda=1, Delay_Period_Log_Normal_Mu=6, Delay_Period_Log_Normal_Sigma=1, Delay_Period_Max=1, Delay_Period_Mean_1=1, Delay_Period_Mean_2=1, Delay_Period_Min=0, Delay_Period_Peak_2_Value=1, Delay_Period_Poisson_Mean=6, Delay_Period_Proportion_0=1, Delay_Period_Proportion_1=1, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='DelayedIntervention', New_Property_Value='', Sim_Types=['*'], iv_type='IndividualTargeted', **kwargs):
        super(DelayedIntervention, self).__init__(**kwargs)
        self.Actual_IndividualIntervention_Configs = Actual_IndividualIntervention_Configs
        self.Coverage = Coverage
        self.Delay_Period_Constant = Delay_Period_Constant
        self.Delay_Period_Distribution = (Delay_Period_Distribution.name if isinstance(Delay_Period_Distribution, Enum) else Delay_Period_Distribution)
        self.Delay_Period_Exponential = Delay_Period_Exponential
        self.Delay_Period_Gaussian_Mean = Delay_Period_Gaussian_Mean
        self.Delay_Period_Gaussian_Std_Dev = Delay_Period_Gaussian_Std_Dev
        self.Delay_Period_Kappa = Delay_Period_Kappa
        self.Delay_Period_Lambda = Delay_Period_Lambda
        self.Delay_Period_Log_Normal_Mu = Delay_Period_Log_Normal_Mu
        self.Delay_Period_Log_Normal_Sigma = Delay_Period_Log_Normal_Sigma
        self.Delay_Period_Max = Delay_Period_Max
        self.Delay_Period_Mean_1 = Delay_Period_Mean_1
        self.Delay_Period_Mean_2 = Delay_Period_Mean_2
        self.Delay_Period_Min = Delay_Period_Min
        self.Delay_Period_Peak_2_Value = Delay_Period_Peak_2_Value
        self.Delay_Period_Poisson_Mean = Delay_Period_Poisson_Mean
        self.Delay_Period_Proportion_0 = Delay_Period_Proportion_0
        self.Delay_Period_Proportion_1 = Delay_Period_Proportion_1
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class DiagnosticTreatNeg(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Defaulters_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': "The intervention configuration given out when an individual did not return for test results when Event_Or_Config is set to 'Config'.",
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Defaulters_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': "Specifies an event that can trigger another intervention when the event occurs. Event_Or_Config must be set to 'Event', and an individual did not return for test results.",
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'DiagnosticTreatNeg',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'If Event_Or_Config is set to Config, this is the intervention given out when there is a negative diagnosis.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Negative_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If an individual tests negative, this specifies an event that may trigger another intervention when the event occurs.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['TBHIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'DiagnosticTreatNeg',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'DiagnosticTreatNeg')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Defaulters_Config=None, Defaulters_Event='', Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Event_Or_Config=DiagnosticTreatNeg_Event_Or_Config_Enum.Config, Intervention_Name='DiagnosticTreatNeg', Negative_Diagnosis_Config=None, Negative_Diagnosis_Event='', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['TBHIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(DiagnosticTreatNeg, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Defaulters_Config = Defaulters_Config
        self.Defaulters_Event = Defaulters_Event
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnosis_Config = Negative_Diagnosis_Config
        self.Negative_Diagnosis_Event = Negative_Diagnosis_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class HIVARTStagingByCD4Diagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'If_Active_TB': {
            'description': "If the individual's CD4 is not below the threshold in the Threshold table and the individual has TB (via their IndividualProperties), then the individual's CD4 will be compared to the CD4 value retrieved from the InterpolatedValueMap matrix based on the current year.",
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'If_Pregnant': {
            'description': "If the individual does not pass the diagnostic from the Threshold or TB matrices, and the individual is pregnant, then the individual's CD4 is compared to the value found in the InterpolatedValueMap matrix.",
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Individual_Property_Active_TB_Key': {
            'default': 'UNINITIALIZED',
            'description': "The IndividualProperty key ('HasActiveTB') used to determine whether the individual has TB.",
            'type': 'string',
        },
        'Individual_Property_Active_TB_Value': {
            'default': 'UNINITIALIZED',
            'description': "The IndividualProperty value ('Yes') used to determine whether the individual has TB.",
            'type': 'string',
        },
        'Intervention_Name': {
            'default': 'HIVARTStagingByCD4Diagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If an individual tests negative, this specifies an event that may trigger another intervention when the event occurs.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['HIV_SIM'],
        'Threshold': {
            'description': "If the individual's CD4 has ever been below the threshold specified, then the test will be positive.",
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'HIVARTStagingByCD4Diagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HIVARTStagingByCD4Diagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, If_Active_TB=None, If_Pregnant=None, Individual_Property_Active_TB_Key='UNINITIALIZED', Individual_Property_Active_TB_Value='UNINITIALIZED', Intervention_Name='HIVARTStagingByCD4Diagnostic', Negative_Diagnosis_Event='', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['HIV_SIM'], Threshold=None, Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(HIVARTStagingByCD4Diagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.If_Active_TB = If_Active_TB
        self.If_Pregnant = If_Pregnant
        self.Individual_Property_Active_TB_Key = Individual_Property_Active_TB_Key
        self.Individual_Property_Active_TB_Value = Individual_Property_Active_TB_Value
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnosis_Event = Negative_Diagnosis_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Threshold = Threshold
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class HIVARTStagingCD4AgnosticDiagnostic(BaseCampaign):
    _definition = {
        'Adult_By_Pregnant': {
            'description': 'Determines the WHO stage at or above which pregnant adults are eligible for ART.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Adult_By_TB': {
            'description': 'Determines the WHO stage at or above which adults having active TB (via individual property Has_Active_TB) are eligible for ART.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Adult_By_WHO_Stage': {
            'description': 'Determines the WHO stage at or above which adults are eligible for ART.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Adult_Treatment_Age': {
            'default': 5,
            'description': 'The age that delineates adult patients from pediatric patients for the purpose of treatment eligibility.',
            'max': 3.40282e+38,
            'min': -1,
            'type': 'float',
        },
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Child_By_TB': {
            'description': 'Determines the WHO stage at or above which children having active TB (via individual property Has_Active_TB) are eligible for ART.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Child_By_WHO_Stage': {
            'description': 'Determines the WHO stage at or above which children are eligible for ART.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Child_Treat_Under_Age_In_Years_Threshold': {
            'description': 'Determines the age at which children are eligible for ART regardless of CD4, WHO stage, or other factors.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Individual_Property_Active_TB_Key': {
            'default': 'UNINITIALIZED',
            'description': "The IndividualProperty key ('HasActiveTB') used to determine whether the individual has TB.",
            'type': 'string',
        },
        'Individual_Property_Active_TB_Value': {
            'default': 'UNINITIALIZED',
            'description': "The IndividualProperty value ('Yes') used to determine whether the individual has TB.",
            'type': 'string',
        },
        'Intervention_Name': {
            'default': 'HIVARTStagingCD4AgnosticDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If an individual tests negative, this specifies an event that may trigger another intervention when the event occurs.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['HIV_SIM', 'TBHIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'HIVARTStagingCD4AgnosticDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HIVARTStagingCD4AgnosticDiagnostic')

    def __init__(self, Adult_By_Pregnant=None, Adult_By_TB=None, Adult_By_WHO_Stage=None, Adult_Treatment_Age=5, Base_Sensitivity=1, Base_Specificity=1, Child_By_TB=None, Child_By_WHO_Stage=None, Child_Treat_Under_Age_In_Years_Threshold=None, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Individual_Property_Active_TB_Key='UNINITIALIZED', Individual_Property_Active_TB_Value='UNINITIALIZED', Intervention_Name='HIVARTStagingCD4AgnosticDiagnostic', Negative_Diagnosis_Event='', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['HIV_SIM', 'TBHIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(HIVARTStagingCD4AgnosticDiagnostic, self).__init__(**kwargs)
        self.Adult_By_Pregnant = Adult_By_Pregnant
        self.Adult_By_TB = Adult_By_TB
        self.Adult_By_WHO_Stage = Adult_By_WHO_Stage
        self.Adult_Treatment_Age = Adult_Treatment_Age
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Child_By_TB = Child_By_TB
        self.Child_By_WHO_Stage = Child_By_WHO_Stage
        self.Child_Treat_Under_Age_In_Years_Threshold = Child_Treat_Under_Age_In_Years_Threshold
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Individual_Property_Active_TB_Key = Individual_Property_Active_TB_Key
        self.Individual_Property_Active_TB_Value = Individual_Property_Active_TB_Value
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnosis_Event = Negative_Diagnosis_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class HIVDelayedIntervention(BaseCampaign):
    _definition = {
        'Broadcast_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The event that should occur at the end of the delay period.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Broadcast_On_Expiration_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the delay intervention expires before arriving at the end of the delay period, this specifies the event that should occur.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Coverage': {
            'default': 1,
            'description': 'The proportion of individuals who receive the DelayedIntervention that actually receive the configured interventions.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Constant': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'The distribution type to use for assigning the delay period for distributing interventions.',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Delay_Period_Exponential': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Kappa': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Lambda': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Delay_Period_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Delay_Period_Max': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Mean_1': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Mean_2': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Min': {
            'default': 0,
            'depends-on': {
                'Delay_Period_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Expiration_Period': {
            'default': 3.40282e+38,
            'description': 'A fixed time period, in days, after which the Broadcast_On_Expiration_Event occurs instead of the Broadcast_Event. Only applied if the Expiration_Period occurs earlier than the end of the delay period.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'HIVDelayedIntervention',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['HIV_SIM'],
        'class': 'HIVDelayedIntervention',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HIVDelayedIntervention')

    def __init__(self, Broadcast_Event='', Broadcast_On_Expiration_Event='', Coverage=1, Delay_Period_Constant=6, Delay_Period_Distribution=HIVDelayedIntervention_Delay_Period_Distribution_Enum.CONSTANT_DISTRIBUTION, Delay_Period_Exponential=6, Delay_Period_Gaussian_Mean=6, Delay_Period_Gaussian_Std_Dev=1, Delay_Period_Kappa=1, Delay_Period_Lambda=1, Delay_Period_Log_Normal_Mu=6, Delay_Period_Log_Normal_Sigma=1, Delay_Period_Max=1, Delay_Period_Mean_1=1, Delay_Period_Mean_2=1, Delay_Period_Min=0, Delay_Period_Peak_2_Value=1, Delay_Period_Poisson_Mean=6, Delay_Period_Proportion_0=1, Delay_Period_Proportion_1=1, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Expiration_Period=3.40282e+38, Intervention_Name='HIVDelayedIntervention', New_Property_Value='', Sim_Types=['HIV_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(HIVDelayedIntervention, self).__init__(**kwargs)
        self.Broadcast_Event = Broadcast_Event
        self.Broadcast_On_Expiration_Event = Broadcast_On_Expiration_Event
        self.Coverage = Coverage
        self.Delay_Period_Constant = Delay_Period_Constant
        self.Delay_Period_Distribution = (Delay_Period_Distribution.name if isinstance(Delay_Period_Distribution, Enum) else Delay_Period_Distribution)
        self.Delay_Period_Exponential = Delay_Period_Exponential
        self.Delay_Period_Gaussian_Mean = Delay_Period_Gaussian_Mean
        self.Delay_Period_Gaussian_Std_Dev = Delay_Period_Gaussian_Std_Dev
        self.Delay_Period_Kappa = Delay_Period_Kappa
        self.Delay_Period_Lambda = Delay_Period_Lambda
        self.Delay_Period_Log_Normal_Mu = Delay_Period_Log_Normal_Mu
        self.Delay_Period_Log_Normal_Sigma = Delay_Period_Log_Normal_Sigma
        self.Delay_Period_Max = Delay_Period_Max
        self.Delay_Period_Mean_1 = Delay_Period_Mean_1
        self.Delay_Period_Mean_2 = Delay_Period_Mean_2
        self.Delay_Period_Min = Delay_Period_Min
        self.Delay_Period_Peak_2_Value = Delay_Period_Peak_2_Value
        self.Delay_Period_Poisson_Mean = Delay_Period_Poisson_Mean
        self.Delay_Period_Proportion_0 = Delay_Period_Proportion_0
        self.Delay_Period_Proportion_1 = Delay_Period_Proportion_1
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Expiration_Period = Expiration_Period
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class HIVDrawBlood(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'HIVDrawBlood',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If an individual tests negative, this specifies an event that may trigger another intervention when the event occurs.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['HIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'HIVDrawBlood',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HIVDrawBlood')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Intervention_Name='HIVDrawBlood', Negative_Diagnosis_Event='', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['HIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(HIVDrawBlood, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnosis_Event = Negative_Diagnosis_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class HIVMuxer(BaseCampaign):
    _definition = {
        'Broadcast_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The event that should occur at the end of the delay period.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Broadcast_On_Expiration_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the delay intervention expires before arriving at the end of the delay period, this specifies the event that should occur.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Coverage': {
            'default': 1,
            'description': 'The proportion of individuals who receive the DelayedIntervention that actually receive the configured interventions.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Constant': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'The distribution type to use for assigning the delay period for distributing interventions.',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Delay_Period_Exponential': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Kappa': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Lambda': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Delay_Period_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Delay_Period_Max': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Mean_1': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Mean_2': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Delay_Period_Min': {
            'default': 0,
            'depends-on': {
                'Delay_Period_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Delay_Period_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Delay_Period_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Delay_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Expiration_Period': {
            'default': 3.40282e+38,
            'description': 'A fixed time period, in days, after which the Broadcast_On_Expiration_Event occurs instead of the Broadcast_Event. Only applied if the Expiration_Period occurs earlier than the end of the delay period.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'HIVMuxer',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Max_Entries': {
            'default': 1,
            'description': 'The maximum number of times the individual can be registered with the HIVMuxer delay. Determines what should happen if an individual reaches the HIVMuxer stage of health care multiple times.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'integer',
        },
        'Muxer_Name': {
            'default': 'UNINITIALIZED STRING',
            'description': 'A name used to identify the delay and check whether individuals have entered it multiple times. If the same name is used at multiple points in the health care process, then the number of entries is combined when Max_Entries is applied.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['HIV_SIM'],
        'class': 'HIVMuxer',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HIVMuxer')

    def __init__(self, Broadcast_Event='', Broadcast_On_Expiration_Event='', Coverage=1, Delay_Period_Constant=6, Delay_Period_Distribution=HIVMuxer_Delay_Period_Distribution_Enum.CONSTANT_DISTRIBUTION, Delay_Period_Exponential=6, Delay_Period_Gaussian_Mean=6, Delay_Period_Gaussian_Std_Dev=1, Delay_Period_Kappa=1, Delay_Period_Lambda=1, Delay_Period_Log_Normal_Mu=6, Delay_Period_Log_Normal_Sigma=1, Delay_Period_Max=1, Delay_Period_Mean_1=1, Delay_Period_Mean_2=1, Delay_Period_Min=0, Delay_Period_Peak_2_Value=1, Delay_Period_Poisson_Mean=6, Delay_Period_Proportion_0=1, Delay_Period_Proportion_1=1, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Expiration_Period=3.40282e+38, Intervention_Name='HIVMuxer', Max_Entries=1, Muxer_Name='UNINITIALIZED STRING', New_Property_Value='', Sim_Types=['HIV_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(HIVMuxer, self).__init__(**kwargs)
        self.Broadcast_Event = Broadcast_Event
        self.Broadcast_On_Expiration_Event = Broadcast_On_Expiration_Event
        self.Coverage = Coverage
        self.Delay_Period_Constant = Delay_Period_Constant
        self.Delay_Period_Distribution = (Delay_Period_Distribution.name if isinstance(Delay_Period_Distribution, Enum) else Delay_Period_Distribution)
        self.Delay_Period_Exponential = Delay_Period_Exponential
        self.Delay_Period_Gaussian_Mean = Delay_Period_Gaussian_Mean
        self.Delay_Period_Gaussian_Std_Dev = Delay_Period_Gaussian_Std_Dev
        self.Delay_Period_Kappa = Delay_Period_Kappa
        self.Delay_Period_Lambda = Delay_Period_Lambda
        self.Delay_Period_Log_Normal_Mu = Delay_Period_Log_Normal_Mu
        self.Delay_Period_Log_Normal_Sigma = Delay_Period_Log_Normal_Sigma
        self.Delay_Period_Max = Delay_Period_Max
        self.Delay_Period_Mean_1 = Delay_Period_Mean_1
        self.Delay_Period_Mean_2 = Delay_Period_Mean_2
        self.Delay_Period_Min = Delay_Period_Min
        self.Delay_Period_Peak_2_Value = Delay_Period_Peak_2_Value
        self.Delay_Period_Poisson_Mean = Delay_Period_Poisson_Mean
        self.Delay_Period_Proportion_0 = Delay_Period_Proportion_0
        self.Delay_Period_Proportion_1 = Delay_Period_Proportion_1
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Expiration_Period = Expiration_Period
        self.Intervention_Name = Intervention_Name
        self.Max_Entries = Max_Entries
        self.Muxer_Name = Muxer_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class HIVPiecewiseByYearAndSexDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Default_Value': {
            'default': 0,
            'description': 'The probability of positive diagnosis if the intervention is used before the earliest specified time in the Time_Value_Map.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Female_Multiplier': {
            'default': 1,
            'description': 'Allows for the probabilities in the Time_Value_Map to be different for males and females, by multiplying the female probabilities by a constant value.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Interpolation_Order': {
            'default': 0,
            'description': "When set to zero, interpolation between values in the Time_Value_Map is zero-order ('staircase'). When set to 1, interpolation between values in the Time_Value_Map is linear. The final value is held constant for all times after the last time specified in the Time_Value_Map.",
            'max': 1,
            'min': 0,
            'type': 'integer',
        },
        'Intervention_Name': {
            'default': 'HIVPiecewiseByYearAndSexDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If an individual tests negative, this specifies an event that may trigger another intervention when the event occurs.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['HIV_SIM'],
        'Time_Value_Map': {
            'description': 'The years (times) and matching probabilities for test results.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'HIVPiecewiseByYearAndSexDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HIVPiecewiseByYearAndSexDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Default_Value=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Female_Multiplier=1, Interpolation_Order=0, Intervention_Name='HIVPiecewiseByYearAndSexDiagnostic', Negative_Diagnosis_Event='', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['HIV_SIM'], Time_Value_Map=None, Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(HIVPiecewiseByYearAndSexDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Default_Value = Default_Value
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Female_Multiplier = Female_Multiplier
        self.Interpolation_Order = Interpolation_Order
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnosis_Event = Negative_Diagnosis_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Time_Value_Map = Time_Value_Map
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class HIVRapidHIVDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'HIVRapidHIVDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If an individual tests negative, this specifies an event that may trigger another intervention when the event occurs.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Probability_Received_Result': {
            'default': 1,
            'description': 'The probability that an individual received the results of a diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['HIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'HIVRapidHIVDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HIVRapidHIVDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Intervention_Name='HIVRapidHIVDiagnostic', Negative_Diagnosis_Event='', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Probability_Received_Result=1, Sim_Types=['HIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(HIVRapidHIVDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnosis_Event = Negative_Diagnosis_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Probability_Received_Result = Probability_Received_Result
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class HIVSigmoidByYearAndSexDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Female_Multiplier': {
            'default': 1,
            'description': 'Allows for the sigmoid time-varying probability to be different for males and females, by multiplying the female probability by a constant value.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'HIVSigmoidByYearAndSexDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If an individual tests negative, this specifies an event that may trigger another intervention when the event occurs.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Ramp_Max': {
            'default': 1,
            'description': 'The right asymptote for the sigmoid trend over time.',
            'max': 1,
            'min': -1,
            'type': 'float',
        },
        'Ramp_MidYear': {
            'default': 2000,
            'description': 'The time of the infection point in the sigmoid trend over time.',
            'max': 2200,
            'min': 1800,
            'type': 'float',
        },
        'Ramp_Min': {
            'default': 0,
            'description': 'The left asymptote for the sigmoid trend over time.',
            'max': 1,
            'min': -1,
            'type': 'float',
        },
        'Ramp_Rate': {
            'default': 1,
            'description': 'The slope of the inflection point in the sigmoid trend over time. A Rate of 1 sets the slope to a 25% change in probability per year.',
            'max': 100,
            'min': -100,
            'type': 'float',
        },
        'Sim_Types': ['HIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'HIVSigmoidByYearAndSexDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HIVSigmoidByYearAndSexDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Female_Multiplier=1, Intervention_Name='HIVSigmoidByYearAndSexDiagnostic', Negative_Diagnosis_Event='', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Ramp_Max=1, Ramp_MidYear=2000, Ramp_Min=0, Ramp_Rate=1, Sim_Types=['HIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(HIVSigmoidByYearAndSexDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Female_Multiplier = Female_Multiplier
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnosis_Event = Negative_Diagnosis_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Ramp_Max = Ramp_Max
        self.Ramp_MidYear = Ramp_MidYear
        self.Ramp_Min = Ramp_Min
        self.Ramp_Rate = Ramp_Rate
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class HIVSimpleDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'HIVSimpleDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If an individual tests negative, this specifies an event that may trigger another intervention when the event occurs.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['HIV_SIM', 'TBHIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'HIVSimpleDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HIVSimpleDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Intervention_Name='HIVSimpleDiagnostic', Negative_Diagnosis_Event='', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['HIV_SIM', 'TBHIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(HIVSimpleDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnosis_Event = Negative_Diagnosis_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class HealthSeekingBehaviorUpdate(BaseCampaign):
    _definition = {
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'HealthSeekingBehaviorUpdate',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'New_Tendency': {
            'default': 1,
            'description': 'The new tendency toward health-seeking behavior that is assigned to the individual, replacing Tendency in HealthSeekingBehaviorUpdateable.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Sim_Types': ['TBHIV_SIM'],
        'class': 'HealthSeekingBehaviorUpdate',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HealthSeekingBehaviorUpdate')

    def __init__(self, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='HealthSeekingBehaviorUpdate', New_Property_Value='', New_Tendency=1, Sim_Types=['TBHIV_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(HealthSeekingBehaviorUpdate, self).__init__(**kwargs)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.New_Tendency = New_Tendency
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class HealthSeekingBehaviorUpdateable(BaseCampaign):
    _definition = {
        'Actual_IndividualIntervention_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The configuration of an actual intervention sought. Selects a class for the intervention and configures the parameters specific for that intervention class.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Actual_IndividualIntervention_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The event of an actual intervention sought. Selects a class for the intervention and configures the parameters specific for that intervention class. See the list of available events for possible values.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'HealthSeekingBehaviorUpdateable',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['TBHIV_SIM'],
        'Single_Use': {
            'default': 1,
            'description': 'If set to true (1), the health-seeking behavior gets used once and discarded. If set to false (0), it remains indefinitely.',
            'type': 'bool',
        },
        'Tendency': {
            'default': 1,
            'description': 'The probability of seeking healthcare.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'HealthSeekingBehaviorUpdateable',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HealthSeekingBehaviorUpdateable')

    def __init__(self, Actual_IndividualIntervention_Config=None, Actual_IndividualIntervention_Event='', Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Event_Or_Config=HealthSeekingBehaviorUpdateable_Event_Or_Config_Enum.Config, Intervention_Name='HealthSeekingBehaviorUpdateable', New_Property_Value='', Sim_Types=['TBHIV_SIM'], Single_Use=True, Tendency=1, iv_type='IndividualTargeted', **kwargs):
        super(HealthSeekingBehaviorUpdateable, self).__init__(**kwargs)
        self.Actual_IndividualIntervention_Config = Actual_IndividualIntervention_Config
        self.Actual_IndividualIntervention_Event = Actual_IndividualIntervention_Event
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Single_Use = Single_Use
        self.Tendency = Tendency
        self.iv_type = iv_type



class HumanHostSeekingTrap(BaseCampaign):
    _definition = {
        'Attract_Config': {
            'description': 'The configuration of attraction efficacy and waning for human host-seeking trap. Decays over time.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 3.75,
            'description': 'Unit cost per trap (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'HumanHostSeekingTrap',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of killing efficacy and waning for human host-seeking trap.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM'],
        'class': 'HumanHostSeekingTrap',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'HumanHostSeekingTrap')

    def __init__(self, Attract_Config=None, Cost_To_Consumer=3.75, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='HumanHostSeekingTrap', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(HumanHostSeekingTrap, self).__init__(**kwargs)
        self.Attract_Config = Attract_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class IRSHousingModification(BaseCampaign):
    _definition = {
        'Blocking_Config': {
            'description': 'The configuration of pre-feed mosquito repellency and waning for housing modification.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 8,
            'description': 'Unit cost per housing modification (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'IRSHousingModification',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of killing efficacy and waning for housing modification. Killing is conditional on NOT blocking the mosquito before feeding.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM'],
        'class': 'IRSHousingModification',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'IRSHousingModification')

    def __init__(self, Blocking_Config=None, Cost_To_Consumer=8, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='IRSHousingModification', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(IRSHousingModification, self).__init__(**kwargs)
        self.Blocking_Config = Blocking_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class IVCalendar(BaseCampaign):
    _definition = {
        'Actual_IndividualIntervention_Configs': {
            'description': 'An array of interventions that will be distributed as specified in the calendar.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Calendar': {
            'default': [],
            'description': 'An array of ages, days and the probabilities of receiving the list of interventions at each age.',
            'item_type': 'AgeAndProbability',
            'type': 'Vector',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Dropout': {
            'default': 0,
            'description': 'If set to true (1), when an intervention distribution is missed, all subsequent interventions are also missed. If set to false (0), all calendar dates/doses are applied independently of each other.',
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'IVCalendar',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['*'],
        'class': 'IVCalendar',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'IVCalendar')

    def __init__(self, Actual_IndividualIntervention_Configs=None, Calendar=[], Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Dropout=False, Intervention_Name='IVCalendar', New_Property_Value='', Sim_Types=['*'], iv_type='IndividualTargeted', **kwargs):
        super(IVCalendar, self).__init__(**kwargs)
        self.Actual_IndividualIntervention_Configs = Actual_IndividualIntervention_Configs
        self.Calendar = Calendar
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Dropout = Dropout
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class ImmunityBloodTest(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'ImmunityBloodTest',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'depends-on': {
                'Event_Or_Config': 'Event',
            },
            'description': 'If an individual tests negative (does not have immunity), then an individual type event is broadcast. This may trigger another intervention when the event occurs. Only used when **Event_Or_Config** is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Positive_Threshold_AcquisitionImmunity': {
            'default': 1,
            'description': 'Specifies the threshold for acquired immunity, where 1 equals 100% immunity and 0 equals 100% susceptible.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['*'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'ImmunityBloodTest',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'ImmunityBloodTest')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Intervention_Name='ImmunityBloodTest', Negative_Diagnosis_Event='', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Positive_Threshold_AcquisitionImmunity=1, Sim_Types=['*'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(ImmunityBloodTest, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnosis_Event = Negative_Diagnosis_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Positive_Threshold_AcquisitionImmunity = Positive_Threshold_AcquisitionImmunity
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class IndividualImmunityChanger(BaseCampaign):
    _definition = {
        'Boost_Acquire': {
            'default': 0,
            'description': 'Specifies the boosting effect on acquisition immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Boost_Mortality': {
            'default': 0,
            'description': 'Specifies the boosting effect on mortality immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Boost_Threshold_Acquire': {
            'default': 0,
            'description': 'Specifies how much acquisition immunity is required before the vaccine changes from a prime to a boost.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Boost_Threshold_Mortality': {
            'default': 0,
            'description': 'Specifies how much mortality immunity is required before the vaccine changes from a prime to a boost.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Boost_Threshold_Transmit': {
            'default': 0,
            'description': 'Specifies how much transmission immunity is required before the vaccine changes from a prime to a boost.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Boost_Transmit': {
            'default': 0,
            'description': 'Specifies the boosting effect on transmission immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': 'Unit cost per vaccine (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Prime_Acquire': {
            'default': 0,
            'description': 'Specifies the priming effect on acquisition immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Prime_Mortality': {
            'default': 0,
            'description': 'Specifies the priming effect on mortality immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Prime_Transmit': {
            'default': 0,
            'description': 'Specifies the priming effect on transmission immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['*'],
        'class': 'IndividualImmunityChanger',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'IndividualImmunityChanger')

    def __init__(self, Boost_Acquire=0, Boost_Mortality=0, Boost_Threshold_Acquire=0, Boost_Threshold_Mortality=0, Boost_Threshold_Transmit=0, Boost_Transmit=0, Cost_To_Consumer=1, Prime_Acquire=0, Prime_Mortality=0, Prime_Transmit=0, Sim_Types=['*'], iv_type='IndividualTargeted', **kwargs):
        super(IndividualImmunityChanger, self).__init__(**kwargs)
        self.Boost_Acquire = Boost_Acquire
        self.Boost_Mortality = Boost_Mortality
        self.Boost_Threshold_Acquire = Boost_Threshold_Acquire
        self.Boost_Threshold_Mortality = Boost_Threshold_Mortality
        self.Boost_Threshold_Transmit = Boost_Threshold_Transmit
        self.Boost_Transmit = Boost_Transmit
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Prime_Acquire = Prime_Acquire
        self.Prime_Mortality = Prime_Mortality
        self.Prime_Transmit = Prime_Transmit
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class InsectKillingFenceHousingModification(BaseCampaign):
    _definition = {
        'Blocking_Config': {
            'description': 'The configuration of pre-feed mosquito repellency and waning for housing modification.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 8,
            'description': 'Unit cost per housing modification (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'InsectKillingFenceHousingModification',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of killing efficacy and waning for housing modification. Killing is conditional on NOT blocking the mosquito before feeding.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM'],
        'class': 'InsectKillingFenceHousingModification',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'InsectKillingFenceHousingModification')

    def __init__(self, Blocking_Config=None, Cost_To_Consumer=8, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='InsectKillingFenceHousingModification', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(InsectKillingFenceHousingModification, self).__init__(**kwargs)
        self.Blocking_Config = Blocking_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class InterventionForCurrentPartners(BaseCampaign):
    _definition = {
        'Broadcast_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'depends-on': {
                'Event_Or_Config': 'Event',
            },
            'description': 'The event that is immediately broadcast to the partner. Required if **Event_or_Config** is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention definition that is immediately distributed to the partner. Required if **Event_Or_Config** is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Intervention_Name': {
            'default': 'InterventionForCurrentPartners',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Maximum_Partners': {
            'default': 100,
            'description': 'The maximum number of partners that will receive the intervention.',
            'max': 100,
            'min': 0,
            'type': 'float',
        },
        'Minimum_Duration_Years': {
            'default': 0,
            'description': 'The minimum amount of time, in years, between relationship formation and the current time for the partner to qualify for the intervention.',
            'max': 200,
            'min': 0,
            'type': 'float',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Prioritize_Partners_By': {
            'default': 'NO_PRIORITIZATION',
            'description': 'How to prioritize partners for the intervention, as long as they have been in a relationship longer than **Minimum_Duration_Years**.',
            'enum': ['NO_PRIORITIZATION', 'CHOSEN_AT_RANDOM', 'LONGER_TIME_IN_RELATIONSHIP', 'SHORTER_TIME_IN_RELATIONSHIP', 'OLDER_AGE', 'YOUNGER_AGE', 'RELATIONSHIP_TYPE'],
            'type': 'enum',
        },
        'Relationship_Types': {
            'default': [],
            'description': 'An array listing all possible relationship types for which partners can qualify for the intervention.',
            'type': 'Vector String',
        },
        'Sim_Types': ['STI_SIM', 'HIV_SIM'],
        'class': 'InterventionForCurrentPartners',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'InterventionForCurrentPartners')

    def __init__(self, Broadcast_Event='', Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Event_Or_Config=InterventionForCurrentPartners_Event_Or_Config_Enum.Config, Intervention_Config=None, Intervention_Name='InterventionForCurrentPartners', Maximum_Partners=100, Minimum_Duration_Years=0, New_Property_Value='', Prioritize_Partners_By=InterventionForCurrentPartners_Prioritize_Partners_By_Enum.NO_PRIORITIZATION, Relationship_Types=[], Sim_Types=['STI_SIM', 'HIV_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(InterventionForCurrentPartners, self).__init__(**kwargs)
        self.Broadcast_Event = Broadcast_Event
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Config = Intervention_Config
        self.Intervention_Name = Intervention_Name
        self.Maximum_Partners = Maximum_Partners
        self.Minimum_Duration_Years = Minimum_Duration_Years
        self.New_Property_Value = New_Property_Value
        self.Prioritize_Partners_By = (Prioritize_Partners_By.name if isinstance(Prioritize_Partners_By, Enum) else Prioritize_Partners_By)
        self.Relationship_Types = Relationship_Types
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class Ivermectin(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 8,
            'description': 'Unit cost per Ivermectin dosing (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'Ivermectin',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of Ivermectin killing efficacy and waning over time.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM'],
        'class': 'Ivermectin',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'Ivermectin')

    def __init__(self, Cost_To_Consumer=8, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='Ivermectin', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(Ivermectin, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class MDRDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Defaulters_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': "The intervention configuration given out when an individual did not return for test results when Event_Or_Config is set to 'Config'.",
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Defaulters_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': "Specifies an event that can trigger another intervention when the event occurs. Event_Or_Config must be set to 'Event', and an individual did not return for test results.",
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'MDRDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'If Event_Or_Config is set to Config, this is the intervention given out when there is a negative diagnosis.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Negative_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If an individual tests negative, this specifies an event that may trigger another intervention when the event occurs.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['TBHIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Treatment_Fraction_Negative_Diagnosis': {
            'default': 1,
            'description': 'The fraction of negative diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'MDRDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'MDRDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Defaulters_Config=None, Defaulters_Event='', Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Event_Or_Config=MDRDiagnostic_Event_Or_Config_Enum.Config, Intervention_Name='MDRDiagnostic', Negative_Diagnosis_Config=None, Negative_Diagnosis_Event='', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['TBHIV_SIM'], Treatment_Fraction=1, Treatment_Fraction_Negative_Diagnosis=1, iv_type='IndividualTargeted', **kwargs):
        super(MDRDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Defaulters_Config = Defaulters_Config
        self.Defaulters_Event = Defaulters_Event
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnosis_Config = Negative_Diagnosis_Config
        self.Negative_Diagnosis_Event = Negative_Diagnosis_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.Treatment_Fraction_Negative_Diagnosis = Treatment_Fraction_Negative_Diagnosis
        self.iv_type = iv_type



class MalariaDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Detection_Threshold': {
            'default': 100,
            'depends-on': {
                'Diagnostic_Type': 'Other',
            },
            'description': 'The diagnostic detection threshold for parasites, in units of microliters of blood.',
            'max': 1000000.0,
            'min': 0,
            'type': 'float',
        },
        'Diagnostic_Type': {
            'default': 'Microscopy',
            'description': 'The type of malaria diagnostic used.',
            'enum': ['Microscopy', 'NewDetectionTech', 'Other'],
            'type': 'enum',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'MalariaDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['MALARIA_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'MalariaDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'MalariaDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Detection_Threshold=100, Diagnostic_Type=MalariaDiagnostic_Diagnostic_Type_Enum.Microscopy, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Event_Or_Config=MalariaDiagnostic_Event_Or_Config_Enum.Config, Intervention_Name='MalariaDiagnostic', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['MALARIA_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(MalariaDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Detection_Threshold = Detection_Threshold
        self.Diagnostic_Type = (Diagnostic_Type.name if isinstance(Diagnostic_Type, Enum) else Diagnostic_Type)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class MaleCircumcision(BaseCampaign):
    _definition = {
        'Apply_If_Higher_Reduced_Acquire': {
            'default': 0,
            'description': 'If set to false (0), the MaleCircumcision intervention can never be applied to someone who already has a MaleCircumcision intervention. If set to true (1), a male who already has a MaleCircumcision intervention, but whose pre-existing MaleCircumcision intervention has a lower efficacy parameter (Circumcision_Reduced_Acquire) than the one about to be applied, will receive the higher-efficacy MaleCircumcision.',
            'type': 'bool',
        },
        'Circumcision_Reduced_Acquire': {
            'default': 0.6,
            'description': 'The reduction of susceptibility to STI by voluntary male medical circumcision (VMMC).',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Distributed_Event_Trigger': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'When defined as part of an intervention block of class MaleCircumcision, this string defines the name of the column in the output files ReportHIVByAgeAndGender.csv and ReportEventRecorder.csv, which log when the intervention has been distributed.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'MaleCircumcision',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['STI_SIM', 'HIV_SIM'],
        'class': 'MaleCircumcision',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'MaleCircumcision')

    def __init__(self, Apply_If_Higher_Reduced_Acquire=False, Circumcision_Reduced_Acquire=0.6, Disqualifying_Properties=[], Distributed_Event_Trigger='', Dont_Allow_Duplicates=False, Intervention_Name='MaleCircumcision', New_Property_Value='', Sim_Types=['STI_SIM', 'HIV_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(MaleCircumcision, self).__init__(**kwargs)
        self.Apply_If_Higher_Reduced_Acquire = Apply_If_Higher_Reduced_Acquire
        self.Circumcision_Reduced_Acquire = Circumcision_Reduced_Acquire
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Distributed_Event_Trigger = Distributed_Event_Trigger
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class MigrateIndividuals(BaseCampaign):
    _definition = {
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Duration_At_Node_Constant': {
            'default': 6,
            'depends-on': {
                'Duration_At_Node_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'The distribution type to use for assigning the duration of time an individual or family spends at a destination node after intervention-based migration.',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Duration_At_Node_Exponential': {
            'default': 6,
            'depends-on': {
                'Duration_At_Node_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Duration_At_Node_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_At_Node_Kappa': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_At_Node_Lambda': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_At_Node_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Duration_At_Node_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Duration_At_Node_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Duration_At_Node_Max': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Mean_1': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_At_Node_Mean_2': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_At_Node_Min': {
            'default': 0,
            'depends-on': {
                'Duration_At_Node_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Duration_At_Node_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Constant': {
            'default': 6,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'The distribution type to use for assigning the duration of time an individual or family waits before migrating to the a destination node after intervention-based migration.',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Duration_Before_Leaving_Exponential': {
            'default': 6,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Kappa': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Lambda': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Max': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Mean_1': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Mean_2': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Min': {
            'default': 0,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'MigrateIndividuals',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Is_Moving': {
            'default': 0,
            'description': 'Set to true (1) to indicate the individual is permanently moving to a new home node for intervention-based migration.',
            'type': 'bool',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'NodeID_To_Migrate_To': {
            'default': 0,
            'description': 'The destination node ID for intervention-based migration.',
            'max': 4294970000.0,
            'min': 0,
            'type': 'integer',
        },
        'Sim_Types': ['*'],
        'class': 'MigrateIndividuals',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'MigrateIndividuals')

    def __init__(self, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Duration_At_Node_Constant=6, Duration_At_Node_Distribution=MigrateIndividuals_Duration_At_Node_Distribution_Enum.CONSTANT_DISTRIBUTION, Duration_At_Node_Exponential=6, Duration_At_Node_Gaussian_Mean=6, Duration_At_Node_Gaussian_Std_Dev=1, Duration_At_Node_Kappa=1, Duration_At_Node_Lambda=1, Duration_At_Node_Log_Normal_Mu=6, Duration_At_Node_Log_Normal_Sigma=1, Duration_At_Node_Max=1, Duration_At_Node_Mean_1=1, Duration_At_Node_Mean_2=1, Duration_At_Node_Min=0, Duration_At_Node_Peak_2_Value=1, Duration_At_Node_Poisson_Mean=6, Duration_At_Node_Proportion_0=1, Duration_At_Node_Proportion_1=1, Duration_Before_Leaving_Constant=6, Duration_Before_Leaving_Distribution=MigrateIndividuals_Duration_Before_Leaving_Distribution_Enum.CONSTANT_DISTRIBUTION, Duration_Before_Leaving_Exponential=6, Duration_Before_Leaving_Gaussian_Mean=6, Duration_Before_Leaving_Gaussian_Std_Dev=1, Duration_Before_Leaving_Kappa=1, Duration_Before_Leaving_Lambda=1, Duration_Before_Leaving_Log_Normal_Mu=6, Duration_Before_Leaving_Log_Normal_Sigma=1, Duration_Before_Leaving_Max=1, Duration_Before_Leaving_Mean_1=1, Duration_Before_Leaving_Mean_2=1, Duration_Before_Leaving_Min=0, Duration_Before_Leaving_Peak_2_Value=1, Duration_Before_Leaving_Poisson_Mean=6, Duration_Before_Leaving_Proportion_0=1, Duration_Before_Leaving_Proportion_1=1, Intervention_Name='MigrateIndividuals', Is_Moving=False, New_Property_Value='', NodeID_To_Migrate_To=0, Sim_Types=['*'], iv_type='IndividualTargeted', **kwargs):
        super(MigrateIndividuals, self).__init__(**kwargs)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Duration_At_Node_Constant = Duration_At_Node_Constant
        self.Duration_At_Node_Distribution = (Duration_At_Node_Distribution.name if isinstance(Duration_At_Node_Distribution, Enum) else Duration_At_Node_Distribution)
        self.Duration_At_Node_Exponential = Duration_At_Node_Exponential
        self.Duration_At_Node_Gaussian_Mean = Duration_At_Node_Gaussian_Mean
        self.Duration_At_Node_Gaussian_Std_Dev = Duration_At_Node_Gaussian_Std_Dev
        self.Duration_At_Node_Kappa = Duration_At_Node_Kappa
        self.Duration_At_Node_Lambda = Duration_At_Node_Lambda
        self.Duration_At_Node_Log_Normal_Mu = Duration_At_Node_Log_Normal_Mu
        self.Duration_At_Node_Log_Normal_Sigma = Duration_At_Node_Log_Normal_Sigma
        self.Duration_At_Node_Max = Duration_At_Node_Max
        self.Duration_At_Node_Mean_1 = Duration_At_Node_Mean_1
        self.Duration_At_Node_Mean_2 = Duration_At_Node_Mean_2
        self.Duration_At_Node_Min = Duration_At_Node_Min
        self.Duration_At_Node_Peak_2_Value = Duration_At_Node_Peak_2_Value
        self.Duration_At_Node_Poisson_Mean = Duration_At_Node_Poisson_Mean
        self.Duration_At_Node_Proportion_0 = Duration_At_Node_Proportion_0
        self.Duration_At_Node_Proportion_1 = Duration_At_Node_Proportion_1
        self.Duration_Before_Leaving_Constant = Duration_Before_Leaving_Constant
        self.Duration_Before_Leaving_Distribution = (Duration_Before_Leaving_Distribution.name if isinstance(Duration_Before_Leaving_Distribution, Enum) else Duration_Before_Leaving_Distribution)
        self.Duration_Before_Leaving_Exponential = Duration_Before_Leaving_Exponential
        self.Duration_Before_Leaving_Gaussian_Mean = Duration_Before_Leaving_Gaussian_Mean
        self.Duration_Before_Leaving_Gaussian_Std_Dev = Duration_Before_Leaving_Gaussian_Std_Dev
        self.Duration_Before_Leaving_Kappa = Duration_Before_Leaving_Kappa
        self.Duration_Before_Leaving_Lambda = Duration_Before_Leaving_Lambda
        self.Duration_Before_Leaving_Log_Normal_Mu = Duration_Before_Leaving_Log_Normal_Mu
        self.Duration_Before_Leaving_Log_Normal_Sigma = Duration_Before_Leaving_Log_Normal_Sigma
        self.Duration_Before_Leaving_Max = Duration_Before_Leaving_Max
        self.Duration_Before_Leaving_Mean_1 = Duration_Before_Leaving_Mean_1
        self.Duration_Before_Leaving_Mean_2 = Duration_Before_Leaving_Mean_2
        self.Duration_Before_Leaving_Min = Duration_Before_Leaving_Min
        self.Duration_Before_Leaving_Peak_2_Value = Duration_Before_Leaving_Peak_2_Value
        self.Duration_Before_Leaving_Poisson_Mean = Duration_Before_Leaving_Poisson_Mean
        self.Duration_Before_Leaving_Proportion_0 = Duration_Before_Leaving_Proportion_0
        self.Duration_Before_Leaving_Proportion_1 = Duration_Before_Leaving_Proportion_1
        self.Intervention_Name = Intervention_Name
        self.Is_Moving = Is_Moving
        self.New_Property_Value = New_Property_Value
        self.NodeID_To_Migrate_To = NodeID_To_Migrate_To
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class ModifyStiCoInfectionStatus(BaseCampaign):
    _definition = {
        'New_STI_CoInfection_Status': {
            'default': 0,
            'description': 'Determines whether to apply STI co-infection, or cure/remove STI co-infection. Set to true (1) to include co-infection; set to false (0) to remove co-infection.',
            'type': 'bool',
        },
        'Sim_Types': ['STI_SIM', 'HIV_SIM'],
        'class': 'ModifyStiCoInfectionStatus',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'ModifyStiCoInfectionStatus')

    def __init__(self, New_STI_CoInfection_Status=False, Sim_Types=['STI_SIM', 'HIV_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(ModifyStiCoInfectionStatus, self).__init__(**kwargs)
        self.New_STI_CoInfection_Status = New_STI_CoInfection_Status
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class MultiEffectBoosterVaccine(BaseCampaign):
    _definition = {
        'Acquire_Config': {
            'description': 'The configuration for multi-effect vaccine acquisition.',
            'type': 'idmType:WaningEffect',
        },
        'Boost_Acquire': {
            'default': 0,
            'description': 'Specifies the boosting effect on acquisition immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Boost_Mortality': {
            'default': 0,
            'description': 'Specifies the boosting effect on mortality immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Boost_Threshold_Acquire': {
            'default': 0,
            'description': 'Specifies how much acquisition immunity is required before the vaccine changes from a prime to a boost.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Boost_Threshold_Mortality': {
            'default': 0,
            'description': 'Specifies how much mortality immunity is required before the vaccine changes from a prime to a boost.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Boost_Threshold_Transmit': {
            'default': 0,
            'description': 'Specifies how much transmission immunity is required before the vaccine changes from a prime to a boost.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Boost_Transmit': {
            'default': 0,
            'description': 'Specifies the boosting effect on transmission immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vaccine (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'MultiEffectBoosterVaccine',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Mortality_Config': {
            'description': 'The configuration for multi-effect vaccine mortality.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Prime_Acquire': {
            'default': 0,
            'description': 'Specifies the priming effect on acquisition immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Prime_Mortality': {
            'default': 0,
            'description': 'Specifies the priming effect on mortality immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Prime_Transmit': {
            'default': 0,
            'description': 'Specifies the priming effect on transmission immunity for naive individuals (without natural or vaccine-derived immunity) for a multi-effect booster vaccine.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['*'],
        'Transmit_Config': {
            'description': 'The configuration for multi-effect vaccine transmission.',
            'type': 'idmType:WaningEffect',
        },
        'Vaccine_Take': {
            'default': 1,
            'description': 'The rate at which delivered vaccines will successfully stimulate an immune response and achieve the desired efficacy.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'MultiEffectBoosterVaccine',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'MultiEffectBoosterVaccine')

    def __init__(self, Acquire_Config=None, Boost_Acquire=0, Boost_Mortality=0, Boost_Threshold_Acquire=0, Boost_Threshold_Mortality=0, Boost_Threshold_Transmit=0, Boost_Transmit=0, Cost_To_Consumer=10, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='MultiEffectBoosterVaccine', Mortality_Config=None, New_Property_Value='', Prime_Acquire=0, Prime_Mortality=0, Prime_Transmit=0, Sim_Types=['*'], Transmit_Config=None, Vaccine_Take=1, iv_type='IndividualTargeted', **kwargs):
        super(MultiEffectBoosterVaccine, self).__init__(**kwargs)
        self.Acquire_Config = Acquire_Config
        self.Boost_Acquire = Boost_Acquire
        self.Boost_Mortality = Boost_Mortality
        self.Boost_Threshold_Acquire = Boost_Threshold_Acquire
        self.Boost_Threshold_Mortality = Boost_Threshold_Mortality
        self.Boost_Threshold_Transmit = Boost_Threshold_Transmit
        self.Boost_Transmit = Boost_Transmit
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Mortality_Config = Mortality_Config
        self.New_Property_Value = New_Property_Value
        self.Prime_Acquire = Prime_Acquire
        self.Prime_Mortality = Prime_Mortality
        self.Prime_Transmit = Prime_Transmit
        self.Sim_Types = Sim_Types
        self.Transmit_Config = Transmit_Config
        self.Vaccine_Take = Vaccine_Take
        self.iv_type = iv_type



class MultiEffectVaccine(BaseCampaign):
    _definition = {
        'Acquire_Config': {
            'description': 'The configuration for multi-effect vaccine acquisition.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vaccine (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'MultiEffectVaccine',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Mortality_Config': {
            'description': 'The configuration for multi-effect vaccine mortality.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['*'],
        'Transmit_Config': {
            'description': 'The configuration for multi-effect vaccine transmission.',
            'type': 'idmType:WaningEffect',
        },
        'Vaccine_Take': {
            'default': 1,
            'description': 'The rate at which delivered vaccines will successfully stimulate an immune response and achieve the desired efficacy.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'MultiEffectVaccine',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'MultiEffectVaccine')

    def __init__(self, Acquire_Config=None, Cost_To_Consumer=10, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='MultiEffectVaccine', Mortality_Config=None, New_Property_Value='', Sim_Types=['*'], Transmit_Config=None, Vaccine_Take=1, iv_type='IndividualTargeted', **kwargs):
        super(MultiEffectVaccine, self).__init__(**kwargs)
        self.Acquire_Config = Acquire_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Mortality_Config = Mortality_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Transmit_Config = Transmit_Config
        self.Vaccine_Take = Vaccine_Take
        self.iv_type = iv_type



class MultiInterventionDistributor(BaseCampaign):
    _definition = {
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_List': {
            'description': 'The list of individual interventions that is distributed by MultiInterventionDistributor.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Intervention_Name': {
            'default': 'MultiInterventionDistributor',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['*'],
        'class': 'MultiInterventionDistributor',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'MultiInterventionDistributor')

    def __init__(self, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_List=None, Intervention_Name='MultiInterventionDistributor', New_Property_Value='', Sim_Types=['*'], iv_type='IndividualTargeted', **kwargs):
        super(MultiInterventionDistributor, self).__init__(**kwargs)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_List = Intervention_List
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class OutbreakIndividual(BaseCampaign):
    _definition = {
        'Antigen': {
            'default': 0,
            'description': 'The antigenic base strain ID of the outbreak infection.',
            'max': 10,
            'min': 0,
            'type': 'integer',
        },
        'Genome': {
            'default': 0,
            'description': 'The genetic substrain ID of the outbreak infection.',
            'max': 16777200.0,
            'min': -1,
            'type': 'integer',
        },
        'Ignore_Immunity': {
            'default': 1,
            'description': 'Individuals will be force-infected (with a specific strain) regardless of actual immunity level when set to true (1).',
            'type': 'bool',
        },
        'Incubation_Period_Override': {
            'default': -1,
            'description': 'The incubation period, in days, that infected individuals will go through before becoming infectious. This value overrides the incubation period set in the configuration file. Set to -1 to honor the configuration parameter settings.',
            'max': 2147480000.0,
            'min': -1,
            'type': 'integer',
        },
        'Sim_Types': ['GENERIC_SIM', 'VECTOR_SIM', 'MALARIA_SIM', 'AIRBORNE_SIM', 'POLIO_SIM', 'TBHIV_SIM', 'STI_SIM', 'HIV_SIM', 'PY_SIM', 'TYPHOID_SIM', 'ENVIRONMENTAL_SIM'],
        'class': 'OutbreakIndividual',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'OutbreakIndividual')

    def __init__(self, Antigen=0, Genome=0, Ignore_Immunity=True, Incubation_Period_Override=-1, Sim_Types=['GENERIC_SIM', 'VECTOR_SIM', 'MALARIA_SIM', 'AIRBORNE_SIM', 'POLIO_SIM', 'TBHIV_SIM', 'STI_SIM', 'HIV_SIM', 'PY_SIM', 'TYPHOID_SIM', 'ENVIRONMENTAL_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(OutbreakIndividual, self).__init__(**kwargs)
        self.Antigen = Antigen
        self.Genome = Genome
        self.Ignore_Immunity = Ignore_Immunity
        self.Incubation_Period_Override = Incubation_Period_Override
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class OutbreakIndividualDengue(BaseCampaign):
    _definition = {
        'Ignore_Immunity': {
            'default': 1,
            'description': 'Individuals will be force-infected (with a specific strain) regardless of actual immunity level when set to true (1).',
            'type': 'bool',
        },
        'Incubation_Period_Override': {
            'default': -1,
            'description': 'The incubation period, in days, that infected individuals will go through before becoming infectious. This value overrides the incubation period set in the configuration file. Set to -1 to honor the configuration parameter settings.',
            'max': 2147480000.0,
            'min': -1,
            'type': 'integer',
        },
        'Sim_Types': ['DENGUE_SIM'],
        'Strain_Id_Name': {
            'default': 'UNINITIALIZED STRING',
            'description': "The Dengue strain to use in this outbreak. E.g., 'Strain_1' or 'Strain_4'.",
            'type': 'string',
        },
        'class': 'OutbreakIndividualDengue',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'OutbreakIndividualDengue')

    def __init__(self, Ignore_Immunity=True, Incubation_Period_Override=-1, Sim_Types=['DENGUE_SIM'], Strain_Id_Name='UNINITIALIZED STRING', iv_type='IndividualTargeted', **kwargs):
        super(OutbreakIndividualDengue, self).__init__(**kwargs)
        self.Ignore_Immunity = Ignore_Immunity
        self.Incubation_Period_Override = Incubation_Period_Override
        self.Sim_Types = Sim_Types
        self.Strain_Id_Name = Strain_Id_Name
        self.iv_type = iv_type



class OutbreakIndividualMalaria(BaseCampaign):
    _definition = {
        'Antigen': {
            'default': 0,
            'description': 'The antigenic base strain ID of the outbreak infection.',
            'max': 10,
            'min': 0,
            'type': 'integer',
        },
        'Create_Random_Genome': {
            'default': 0,
            'description': 'If set to true (1), then a random genome is created for the infection and the Genome_Markers parameter is not used. If set to false (0), then you must define the  Genome_Markers parameter which allows you to then specify genetic components in a strain of infection.',
            'type': 'bool',
        },
        'Genome_Markers': {
            'default': [],
            'description': 'A list of the names of genome marker(s) that represent the genetic components in a strain of an infection.',
            'type': 'Vector String',
            'value_source': '<configuration>.Genome_Markers',
        },
        'Ignore_Immunity': {
            'default': 1,
            'description': 'Individuals will be force-infected (with a specific strain) regardless of actual immunity level when set to true (1).',
            'type': 'bool',
        },
        'Incubation_Period_Override': {
            'default': -1,
            'description': 'The incubation period, in days, that infected individuals will go through before becoming infectious. This value overrides the incubation period set in the configuration file. Set to -1 to honor the configuration parameter settings.',
            'max': 2147480000.0,
            'min': -1,
            'type': 'integer',
        },
        'Sim_Types': ['MALARIA_SIM'],
        'class': 'OutbreakIndividualMalaria',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'OutbreakIndividualMalaria')

    def __init__(self, Antigen=0, Create_Random_Genome=False, Genome_Markers=[], Ignore_Immunity=True, Incubation_Period_Override=-1, Sim_Types=['MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(OutbreakIndividualMalaria, self).__init__(**kwargs)
        self.Antigen = Antigen
        self.Create_Random_Genome = Create_Random_Genome
        self.Genome_Markers = Genome_Markers
        self.Ignore_Immunity = Ignore_Immunity
        self.Incubation_Period_Override = Incubation_Period_Override
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class OutbreakIndividualTBorHIV(BaseCampaign):
    _definition = {
        'Antigen': {
            'default': 0,
            'description': 'The antigenic base strain ID of the outbreak infection.',
            'max': 10,
            'min': 0,
            'type': 'integer',
        },
        'Genome': {
            'default': 0,
            'description': 'The genetic substrain ID of the outbreak infection.',
            'max': 16777200.0,
            'min': -1,
            'type': 'integer',
        },
        'Ignore_Immunity': {
            'default': 1,
            'description': 'Individuals will be force-infected (with a specific strain) regardless of actual immunity level when set to true (1).',
            'type': 'bool',
        },
        'Incubation_Period_Override': {
            'default': -1,
            'description': 'The incubation period, in days, that infected individuals will go through before becoming infectious. This value overrides the incubation period set in the configuration file. Set to -1 to honor the configuration parameter settings.',
            'max': 2147480000.0,
            'min': -1,
            'type': 'integer',
        },
        'Infection_Type': {
            'default': 'HIV',
            'description': 'TB =1 or HIV = 0',
            'enum': ['HIV', 'TB'],
            'type': 'enum',
        },
        'Sim_Types': ['TBHIV_SIM'],
        'class': 'OutbreakIndividualTBorHIV',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'OutbreakIndividualTBorHIV')

    def __init__(self, Antigen=0, Genome=0, Ignore_Immunity=True, Incubation_Period_Override=-1, Infection_Type=OutbreakIndividualTBorHIV_Infection_Type_Enum.HIV, Sim_Types=['TBHIV_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(OutbreakIndividualTBorHIV, self).__init__(**kwargs)
        self.Antigen = Antigen
        self.Genome = Genome
        self.Ignore_Immunity = Ignore_Immunity
        self.Incubation_Period_Override = Incubation_Period_Override
        self.Infection_Type = (Infection_Type.name if isinstance(Infection_Type, Enum) else Infection_Type)
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class PMTCT(BaseCampaign):
    _definition = {
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Efficacy': {
            'default': 0.5,
            'description': 'Represents the efficacy of a Prevention of Mother to Child Transmission (PMTCT) intervention, defined as the rate ratio of mother to child transmission (MTCT) between women receiving the intervention and women not receiving the intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'PMTCT',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['HIV_SIM'],
        'class': 'PMTCT',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'PMTCT')

    def __init__(self, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Efficacy=0.5, Intervention_Name='PMTCT', New_Property_Value='', Sim_Types=['HIV_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(PMTCT, self).__init__(**kwargs)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Efficacy = Efficacy
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class PolioVaccine(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Cost of each polio vaccine.',
            'max': 100,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'PolioVaccine',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['POLIO_SIM'],
        'Time_Since_Vaccination': {
            'default': 0,
            'description': 'Time since vaccination.',
            'max': 100,
            'min': 0,
            'type': 'float',
        },
        'Vaccine_Type': {
            'default': 'tOPV',
            'description': 'Polio vaccine type.',
            'enum': ['tOPV', 'bOPV', 'mOPV_1', 'mOPV_2', 'mOPV_3', 'eIPV'],
            'type': 'enum',
        },
        'class': 'PolioVaccine',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'PolioVaccine')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='PolioVaccine', New_Property_Value='', Sim_Types=['POLIO_SIM'], Time_Since_Vaccination=0, Vaccine_Type=PolioVaccine_Vaccine_Type_Enum.tOPV, iv_type='IndividualTargeted', **kwargs):
        super(PolioVaccine, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Time_Since_Vaccination = Time_Since_Vaccination
        self.Vaccine_Type = (Vaccine_Type.name if isinstance(Vaccine_Type, Enum) else Vaccine_Type)
        self.iv_type = iv_type



class PropertyValueChanger(BaseCampaign):
    _definition = {
        'Daily_Probability': {
            'default': 1,
            'description': 'The daily probability that an individual will move to the Target_Property_Value.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'PropertyValueChanger',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Maximum_Duration': {
            'default': 3.40282e+38,
            'description': 'The maximum amount of time individuals have to move to a new group. This timing works in conjunction with Daily_Probability.',
            'max': 3.40282e+38,
            'min': -1,
            'type': 'float',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Revert': {
            'default': 0,
            'description': 'The number of days before an individual moves back to their original group.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['*'],
        'Target_Property_Key': {
            'default': 'UNINITIALIZED STRING',
            'description': 'The name of the individual property type whose value will be updated by the intervention.',
            'type': 'Constrained String',
            'value_source': '<demographics>::*.Individual_Properties.*.Property',
        },
        'Target_Property_Value': {
            'default': 'UNINITIALIZED STRING',
            'description': 'The user-defined value of the individual property that will be assigned to the individual.',
            'type': 'Constrained String',
            'value_source': '<demographics>::*.Individual_Properties.*.Values',
        },
        'class': 'PropertyValueChanger',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'PropertyValueChanger')

    def __init__(self, Daily_Probability=1, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='PropertyValueChanger', Maximum_Duration=3.40282e+38, New_Property_Value='', Revert=0, Sim_Types=['*'], Target_Property_Key='UNINITIALIZED STRING', Target_Property_Value='UNINITIALIZED STRING', iv_type='IndividualTargeted', **kwargs):
        super(PropertyValueChanger, self).__init__(**kwargs)
        self.Daily_Probability = Daily_Probability
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Maximum_Duration = Maximum_Duration
        self.New_Property_Value = New_Property_Value
        self.Revert = Revert
        self.Sim_Types = Sim_Types
        self.Target_Property_Key = Target_Property_Key
        self.Target_Property_Value = Target_Property_Value
        self.iv_type = iv_type



class RTSSVaccine(BaseCampaign):
    _definition = {
        'Antibody_Type': {
            'default': 'CSP',
            'description': 'The antibody type used to determine immune response targets.',
            'enum': ['CSP', 'MSP1', 'PfEMP1_minor', 'PfEMP1_major', 'N_MALARIA_ANTIBODY_TYPES'],
            'type': 'enum',
        },
        'Antibody_Variant': {
            'default': 0,
            'description': 'The antibody variant index.',
            'max': 100000,
            'min': 0,
            'type': 'integer',
        },
        'Boosted_Antibody_Concentration': {
            'default': 1,
            'description': 'The boosted antibody concentration, where unity equals maximum from natural exposure.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 3.75,
            'description': 'Unit cost of RTS,S vaccination (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'RTSSVaccine',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['MALARIA_SIM'],
        'class': 'RTSSVaccine',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'RTSSVaccine')

    def __init__(self, Antibody_Type=RTSSVaccine_Antibody_Type_Enum.CSP, Antibody_Variant=0, Boosted_Antibody_Concentration=1, Cost_To_Consumer=3.75, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='RTSSVaccine', New_Property_Value='', Sim_Types=['MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(RTSSVaccine, self).__init__(**kwargs)
        self.Antibody_Type = (Antibody_Type.name if isinstance(Antibody_Type, Enum) else Antibody_Type)
        self.Antibody_Variant = Antibody_Variant
        self.Boosted_Antibody_Concentration = Boosted_Antibody_Concentration
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class RandomChoice(BaseCampaign):
    _definition = {
        'Choice_Names': {
            'default': [],
            'description': 'An array of event names to be broadcast if randomly selected, used with Choice_Probabilities.',
            'type': 'Vector String',
        },
        'Choice_Probabilities': {
            'ascending': 0,
            'default': [],
            'description': 'An array of probabilities that the event will be selected, used with Choice_Names. Values in map will be normalized.',
            'max': 1,
            'min': 0,
            'type': 'Vector Float',
        },
        'Cost_To_Consumer': {
            'default': 0,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'RandomChoice',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['*'],
        'class': 'RandomChoice',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'RandomChoice')

    def __init__(self, Choice_Names=[], Choice_Probabilities=[], Cost_To_Consumer=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='RandomChoice', New_Property_Value='', Sim_Types=['*'], iv_type='IndividualTargeted', **kwargs):
        super(RandomChoice, self).__init__(**kwargs)
        self.Choice_Names = Choice_Names
        self.Choice_Probabilities = Choice_Probabilities
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class RandomChoiceMatrix(BaseCampaign):
    _definition = {
        'Choices': {
            'default': [],
            'description': 'TBD',
            'item_type': 'Choice',
            'type': 'Vector',
        },
        'Filters': {
            'default': [],
            'description': 'TBD',
            'item_type': 'Filter',
            'type': 'Vector',
        },
        'Sim_Types': ['FP_SIM'],
        'class': 'RandomChoiceMatrix',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'RandomChoiceMatrix')

    def __init__(self, Choices=[], Filters=[], Sim_Types=['FP_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(RandomChoiceMatrix, self).__init__(**kwargs)
        self.Choices = Choices
        self.Filters = Filters
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class STIBarrier(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Determines the unit cost when using the STIBarrier intervention to change defaults from demographics. Note that there is no cost for condoms distributed using demographics-configured default usage probabilities.',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Early': {
            'default': 1,
            'description': 'The left asymptote for the sigmoid trend over time. The Early value must be smaller than the Late value.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'STIBarrier',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Late': {
            'default': 1,
            'description': 'The right asymptote for the sigmoid trend over time. The Late value must be larger than the Early value.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'MidYear': {
            'default': 2000,
            'description': 'The time of the inflection point in the sigmoid trend over time.',
            'max': 2200,
            'min': 1800,
            'type': 'float',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Rate': {
            'default': 1,
            'description': 'The slope of the inflection point in the sigmoid trend over time. A Rate of 1 sets the slope to a 25% change in probability per year. Specify a negative Rate (e.g. -1) to achieve a negative sigmoid.',
            'max': 100,
            'min': -100,
            'type': 'float',
        },
        'Relationship_Type': {
            'default': 'TRANSITORY',
            'description': 'The relationship type to which the condom usage probability is applied.',
            'enum': ['TRANSITORY', 'INFORMAL', 'MARITAL', 'COMMERCIAL', 'COUNT'],
            'type': 'enum',
        },
        'Sim_Types': ['STI_SIM', 'HIV_SIM'],
        'class': 'STIBarrier',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'STIBarrier')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Early=1, Intervention_Name='STIBarrier', Late=1, MidYear=2000, New_Property_Value='', Rate=1, Relationship_Type=STIBarrier_Relationship_Type_Enum.TRANSITORY, Sim_Types=['STI_SIM', 'HIV_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(STIBarrier, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Early = Early
        self.Intervention_Name = Intervention_Name
        self.Late = Late
        self.MidYear = MidYear
        self.New_Property_Value = New_Property_Value
        self.Rate = Rate
        self.Relationship_Type = (Relationship_Type.name if isinstance(Relationship_Type, Enum) else Relationship_Type)
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class STIIsPostDebut(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'STIIsPostDebut',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The name of the event to broadcast when an individual is found to NOT be Post-Debut age.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['STI_SIM', 'HIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'STIIsPostDebut',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'STIIsPostDebut')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Event_Or_Config=STIIsPostDebut_Event_Or_Config_Enum.Config, Intervention_Name='STIIsPostDebut', Negative_Diagnosis_Event='', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['STI_SIM', 'HIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(STIIsPostDebut, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnosis_Event = Negative_Diagnosis_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class ScreeningHousingModification(BaseCampaign):
    _definition = {
        'Blocking_Config': {
            'description': 'The configuration of pre-feed mosquito repellency and waning for housing modification.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 8,
            'description': 'Unit cost per housing modification (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'ScreeningHousingModification',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of killing efficacy and waning for housing modification. Killing is conditional on NOT blocking the mosquito before feeding.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM'],
        'class': 'ScreeningHousingModification',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'ScreeningHousingModification')

    def __init__(self, Blocking_Config=None, Cost_To_Consumer=8, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='ScreeningHousingModification', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(ScreeningHousingModification, self).__init__(**kwargs)
        self.Blocking_Config = Blocking_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class SimpleBednet(BaseCampaign):
    _definition = {
        'Blocking_Config': {
            'description': 'Configures the rate of blocking for indoor mosquito feeds on individuals with an ITN; decays over time.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 3.75,
            'description': 'Unit cost per bednet (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'SimpleBednet',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of the rate at which mosquitoes die, conditional on a successfully blocked feed; decays over time.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['MALARIA_SIM', 'VECTOR_SIM'],
        'Usage_Config': {
            'description': 'The user-defined WaningEffects to determine when and if an individual is using a bed net.',
            'type': 'idmType:WaningEffect',
        },
        'class': 'SimpleBednet',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'SimpleBednet')

    def __init__(self, Blocking_Config=None, Cost_To_Consumer=3.75, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='SimpleBednet', Killing_Config=None, New_Property_Value='', Sim_Types=['MALARIA_SIM', 'VECTOR_SIM'], Usage_Config=None, iv_type='IndividualTargeted', **kwargs):
        super(SimpleBednet, self).__init__(**kwargs)
        self.Blocking_Config = Blocking_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Usage_Config = Usage_Config
        self.iv_type = iv_type



class SimpleBoosterVaccine(BaseCampaign):
    _definition = {
        'Boost_Effect': {
            'default': 1,
            'description': 'Specifies the boosting effect on [acquisition/transmission/mortality] immunity for previously exposed individuals (either natural or vaccine-derived).',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Boost_Threshold': {
            'default': 0,
            'description': 'Specifies how much immunity is required before the vaccine changes from a priming effect to a boosting effect.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vaccine (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Efficacy_Is_Multiplicative': {
            'default': 1,
            'description': 'The overall vaccine efficacy when individuals receive more than one vaccine. When set to true (1), the vaccine efficacies are multiplied together; when set to false (0), the efficacies are additive.',
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'SimpleBoosterVaccine',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Prime_Effect': {
            'default': 1,
            'description': 'Specifies the priming effect on [acquisition/transmission/mortality] immunity for naive individuals (without natural or vaccine-derived immunity).',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['*'],
        'Vaccine_Take': {
            'default': 1,
            'description': 'The rate at which delivered vaccines will successfully stimulate an immune response and achieve the desired efficacy.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Vaccine_Type': {
            'default': 'Generic',
            'description': 'The type of vaccine to distribute in a vaccine intervention.',
            'enum': ['Generic', 'TransmissionBlocking', 'AcquisitionBlocking', 'MortalityBlocking'],
            'type': 'enum',
        },
        'Waning_Config': {
            'description': 'The configuration of Ivermectin killing efficacy and waning over time.',
            'type': 'idmType:WaningEffect',
        },
        'class': 'SimpleBoosterVaccine',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'SimpleBoosterVaccine')

    def __init__(self, Boost_Effect=1, Boost_Threshold=0, Cost_To_Consumer=10, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Efficacy_Is_Multiplicative=True, Intervention_Name='SimpleBoosterVaccine', New_Property_Value='', Prime_Effect=1, Sim_Types=['*'], Vaccine_Take=1, Vaccine_Type=SimpleBoosterVaccine_Vaccine_Type_Enum.Generic, Waning_Config=[], iv_type='IndividualTargeted', **kwargs):
        super(SimpleBoosterVaccine, self).__init__(**kwargs)
        self.Boost_Effect = Boost_Effect
        self.Boost_Threshold = Boost_Threshold
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Efficacy_Is_Multiplicative = Efficacy_Is_Multiplicative
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Prime_Effect = Prime_Effect
        self.Sim_Types = Sim_Types
        self.Vaccine_Take = Vaccine_Take
        self.Vaccine_Type = (Vaccine_Type.name if isinstance(Vaccine_Type, Enum) else Vaccine_Type)
        self.Waning_Config = Waning_Config
        self.iv_type = iv_type



class SimpleDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'SimpleDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['*'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'SimpleDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'SimpleDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Event_Or_Config=SimpleDiagnostic_Event_Or_Config_Enum.Config, Intervention_Name='SimpleDiagnostic', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['*'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(SimpleDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class SimpleHealthSeekingBehavior(BaseCampaign):
    _definition = {
        'Actual_IndividualIntervention_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The configuration of an actual intervention sought. Selects a class for the intervention and configures the parameters specific for that intervention class.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Actual_IndividualIntervention_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The event of an actual intervention sought. Selects a class for the intervention and configures the parameters specific for that intervention class. See the list of available events for possible values.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'SimpleHealthSeekingBehavior',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['*'],
        'Single_Use': {
            'default': 1,
            'description': 'If set to true (1), the health-seeking behavior gets used once and discarded. If set to false (0), it remains indefinitely.',
            'type': 'bool',
        },
        'Tendency': {
            'default': 1,
            'description': 'The probability of seeking healthcare.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'SimpleHealthSeekingBehavior',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'SimpleHealthSeekingBehavior')

    def __init__(self, Actual_IndividualIntervention_Config=None, Actual_IndividualIntervention_Event='', Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Event_Or_Config=SimpleHealthSeekingBehavior_Event_Or_Config_Enum.Config, Intervention_Name='SimpleHealthSeekingBehavior', New_Property_Value='', Sim_Types=['*'], Single_Use=True, Tendency=1, iv_type='IndividualTargeted', **kwargs):
        super(SimpleHealthSeekingBehavior, self).__init__(**kwargs)
        self.Actual_IndividualIntervention_Config = Actual_IndividualIntervention_Config
        self.Actual_IndividualIntervention_Event = Actual_IndividualIntervention_Event
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Single_Use = Single_Use
        self.Tendency = Tendency
        self.iv_type = iv_type



class SimpleHousingModification(BaseCampaign):
    _definition = {
        'Blocking_Config': {
            'description': 'The configuration of pre-feed mosquito repellency and waning for housing modification.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 8,
            'description': 'Unit cost per housing modification (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'SimpleHousingModification',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of killing efficacy and waning for housing modification. Killing is conditional on NOT blocking the mosquito before feeding.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM'],
        'class': 'SimpleHousingModification',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'SimpleHousingModification')

    def __init__(self, Blocking_Config=None, Cost_To_Consumer=8, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='SimpleHousingModification', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(SimpleHousingModification, self).__init__(**kwargs)
        self.Blocking_Config = Blocking_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class SimpleIndividualRepellent(BaseCampaign):
    _definition = {
        'Blocking_Config': {
            'description': 'The configuration of efficacy and waning for individual repellent.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 8,
            'description': 'Unit cost per repellent (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'SimpleIndividualRepellent',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM'],
        'class': 'SimpleIndividualRepellent',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'SimpleIndividualRepellent')

    def __init__(self, Blocking_Config=None, Cost_To_Consumer=8, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='SimpleIndividualRepellent', New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(SimpleIndividualRepellent, self).__init__(**kwargs)
        self.Blocking_Config = Blocking_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class SimpleVaccine(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vaccine (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Efficacy_Is_Multiplicative': {
            'default': 1,
            'description': 'The overall vaccine efficacy when individuals receive more than one vaccine. When set to true (1), the vaccine efficacies are multiplied together; when set to false (0), the efficacies are additive.',
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'SimpleVaccine',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['*'],
        'Vaccine_Take': {
            'default': 1,
            'description': 'The rate at which delivered vaccines will successfully stimulate an immune response and achieve the desired efficacy.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Vaccine_Type': {
            'default': 'Generic',
            'description': 'The type of vaccine to distribute in a vaccine intervention.',
            'enum': ['Generic', 'TransmissionBlocking', 'AcquisitionBlocking', 'MortalityBlocking'],
            'type': 'enum',
        },
        'Waning_Config': {
            'description': 'The configuration of Ivermectin killing efficacy and waning over time.',
            'type': 'idmType:WaningEffect',
        },
        'class': 'SimpleVaccine',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'SimpleVaccine')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Efficacy_Is_Multiplicative=True, Intervention_Name='SimpleVaccine', New_Property_Value='', Sim_Types=['*'], Vaccine_Take=1, Vaccine_Type=SimpleVaccine_Vaccine_Type_Enum.Generic, Waning_Config=[], iv_type='IndividualTargeted', **kwargs):
        super(SimpleVaccine, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Efficacy_Is_Multiplicative = Efficacy_Is_Multiplicative
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Vaccine_Take = Vaccine_Take
        self.Vaccine_Type = (Vaccine_Type.name if isinstance(Vaccine_Type, Enum) else Vaccine_Type)
        self.Waning_Config = Waning_Config
        self.iv_type = iv_type



class SmearDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'SmearDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['TBHIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'SmearDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'SmearDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Event_Or_Config=SmearDiagnostic_Event_Or_Config_Enum.Config, Intervention_Name='SmearDiagnostic', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['TBHIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(SmearDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class SpatialRepellentHousingModification(BaseCampaign):
    _definition = {
        'Blocking_Config': {
            'description': 'The configuration of pre-feed mosquito repellency and waning for housing modification.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 8,
            'description': 'Unit cost per housing modification (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Intervention_Name': {
            'default': 'SpatialRepellentHousingModification',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of killing efficacy and waning for housing modification. Killing is conditional on NOT blocking the mosquito before feeding.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM'],
        'class': 'SpatialRepellentHousingModification',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'SpatialRepellentHousingModification')

    def __init__(self, Blocking_Config=None, Cost_To_Consumer=8, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Intervention_Name='SpatialRepellentHousingModification', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(SpatialRepellentHousingModification, self).__init__(**kwargs)
        self.Blocking_Config = Blocking_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class StiCoInfectionDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'StiCoInfectionDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['STI_SIM', 'HIV_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'StiCoInfectionDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'StiCoInfectionDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Event_Or_Config=StiCoInfectionDiagnostic_Event_Or_Config_Enum.Config, Intervention_Name='StiCoInfectionDiagnostic', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['STI_SIM', 'HIV_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(StiCoInfectionDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class TBHIVConfigurableTBdrug(BaseCampaign):
    _definition = {
        'Active_Multiplier': {
            'default': 1,
            'description': 'Multiplier of clearance/inactivation if active TB on drug treatment.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': 'Unit cost per drug (unamortized).',
            'max': 99999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Dose_Interval': {
            'default': 1,
            'description': 'The interval between doses, in days.',
            'max': 99999,
            'min': 0,
            'type': 'float',
        },
        'Drug_CMax': {
            'default': 1,
            'description': 'The maximum drug concentration that can be used, and is in the same units as Drug_PKPD_C50.',
            'max': 10000,
            'min': 0,
            'type': 'float',
        },
        'Drug_PKPD_C50': {
            'default': 1,
            'description': 'The concentration at which drug killing rates are half of the maximum. Must use the same units as Drug_Cmax.',
            'max': 5000,
            'min': 0,
            'type': 'float',
        },
        'Drug_Vd': {
            'default': 1,
            'description': 'The volume of drug distribution. This value is the ratio of the volume of the second compartment to the volume of the first compartment in a two-compartment model, and is dimensionless.',
            'max': 10000,
            'min': 0,
            'type': 'float',
        },
        'Durability_Profile': {
            'default': 'FIXED_DURATION_CONSTANT_EFFECT',
            'description': 'The profile of durability decay.',
            'enum': ['FIXED_DURATION_CONSTANT_EFFECT', 'CONCENTRATION_VERSUS_TIME'],
            'type': 'enum',
        },
        'Fraction_Defaulters': {
            'default': 0,
            'description': 'The fraction of individuals who will not finish their drug treatment.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'TBHIVConfigurableTBdrug',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Latency_Multiplier': {
            'default': 1,
            'description': 'Multiplier of clearance/inactivation if latent TB on drug treatment.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Primary_Decay_Time_Constant': {
            'default': 1,
            'description': 'The primary decay time constant (in days) of the decay profile.',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Remaining_Doses': {
            'default': 0,
            'description': 'The remaining doses in an intervention; enter a negative number for unlimited doses.',
            'max': 999999,
            'min': -1,
            'type': 'integer',
        },
        'Secondary_Decay_Time_Constant': {
            'default': 1,
            'description': 'The secondary decay time constant of the durability profile.',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['TBHIV_SIM'],
        'TB_Drug_Name': {
            'default': 'UNINITIALIZED STRING',
            'description': 'No Description Yet',
            'type': 'string',
        },
        'class': 'TBHIVConfigurableTBdrug',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'TBHIVConfigurableTBdrug')

    def __init__(self, Active_Multiplier=1, Cost_To_Consumer=1, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Dose_Interval=1, Drug_CMax=1, Drug_PKPD_C50=1, Drug_Vd=1, Durability_Profile=TBHIVConfigurableTBdrug_Durability_Profile_Enum.FIXED_DURATION_CONSTANT_EFFECT, Fraction_Defaulters=0, Intervention_Name='TBHIVConfigurableTBdrug', Latency_Multiplier=1, New_Property_Value='', Primary_Decay_Time_Constant=1, Remaining_Doses=0, Secondary_Decay_Time_Constant=1, Sim_Types=['TBHIV_SIM'], TB_Drug_Name='UNINITIALIZED STRING', iv_type='IndividualTargeted', **kwargs):
        super(TBHIVConfigurableTBdrug, self).__init__(**kwargs)
        self.Active_Multiplier = Active_Multiplier
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Dose_Interval = Dose_Interval
        self.Drug_CMax = Drug_CMax
        self.Drug_PKPD_C50 = Drug_PKPD_C50
        self.Drug_Vd = Drug_Vd
        self.Durability_Profile = (Durability_Profile.name if isinstance(Durability_Profile, Enum) else Durability_Profile)
        self.Fraction_Defaulters = Fraction_Defaulters
        self.Intervention_Name = Intervention_Name
        self.Latency_Multiplier = Latency_Multiplier
        self.New_Property_Value = New_Property_Value
        self.Primary_Decay_Time_Constant = Primary_Decay_Time_Constant
        self.Remaining_Doses = Remaining_Doses
        self.Secondary_Decay_Time_Constant = Secondary_Decay_Time_Constant
        self.Sim_Types = Sim_Types
        self.TB_Drug_Name = TB_Drug_Name
        self.iv_type = iv_type



class TyphoidCarrierClear(BaseCampaign):
    _definition = {
        'Clearance_Rate': {
            'default': 1,
            'description': '',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['TYPHOID_SIM'],
        'class': 'TyphoidCarrierClear',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'TyphoidCarrierClear')

    def __init__(self, Clearance_Rate=1, Sim_Types=['TYPHOID_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(TyphoidCarrierClear, self).__init__(**kwargs)
        self.Clearance_Rate = Clearance_Rate
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class TyphoidCarrierDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The sensitivity of the diagnostic. This sets the proportion of the time that individuals with the condition being tested receive a positive diagnostic test. When set to zero, then individuals who have the condition always receive a false-negative diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Cost_To_Consumer': {
            'default': 1,
            'description': "The unit 'cost' assigned to the diagnostic. Setting Cost_To_Consumer to zero for all other interventions, and to a non-zero amount for one intervention, provides a convenient way to track the number of times the intervention has been applied in a simulation.",
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Days_To_Diagnosis': {
            'default': 0,
            'description': 'The number of days from test until diagnosis.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Enable_IsSymptomatic': {
            'default': 0,
            'description': 'If true, requires an infection to be symptomatic to return a positive test.',
            'type': 'bool',
        },
        'Event_Or_Config': {
            'default': 'Config',
            'description': 'Specifies whether the current intervention (or a positive diagnosis, depending on the intervention class) distributes a nested intervention (the Config option) or an event will be broadcast which may trigger other interventions in the campaign file (the Event option).',
            'enum': ['Config', 'Event'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'TyphoidCarrierDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Positive_Diagnosis_Config': {
            'depends-on': {
                'Event_Or_Config': 'Config',
            },
            'description': 'The intervention distributed to individuals if they test positive. Only used when Event_Or_Config is set to Config.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Positive_Diagnosis_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'If the test is positive, this specifies an event that can trigger another intervention when the event occurs. Only used if Event_Or_Config is set to Event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['TYPHOID_SIM'],
        'Treatment_Fraction': {
            'default': 1,
            'description': 'The fraction of positive diagnoses that are treated.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'TyphoidCarrierDiagnostic',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'TyphoidCarrierDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Cost_To_Consumer=1, Days_To_Diagnosis=0, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Enable_IsSymptomatic=False, Event_Or_Config=TyphoidCarrierDiagnostic_Event_Or_Config_Enum.Config, Intervention_Name='TyphoidCarrierDiagnostic', New_Property_Value='', Positive_Diagnosis_Config=None, Positive_Diagnosis_Event='', Sim_Types=['TYPHOID_SIM'], Treatment_Fraction=1, iv_type='IndividualTargeted', **kwargs):
        super(TyphoidCarrierDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Days_To_Diagnosis = Days_To_Diagnosis
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Enable_IsSymptomatic = Enable_IsSymptomatic
        self.Event_Or_Config = (Event_Or_Config.name if isinstance(Event_Or_Config, Enum) else Event_Or_Config)
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnosis_Config = Positive_Diagnosis_Config
        self.Positive_Diagnosis_Event = Positive_Diagnosis_Event
        self.Sim_Types = Sim_Types
        self.Treatment_Fraction = Treatment_Fraction
        self.iv_type = iv_type



class TyphoidVaccine(BaseCampaign):
    _definition = {
        'Changing_Effect': {
            'description': 'A highly configurable effect that changes over time.',
            'type': 'idmType:WaningEffect',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Effect': {
            'default': 1,
            'description': 'The efficacy of the Typhoid vaccine intervention. For example, a value of 1 would be 100 percent efficacy for all targeted nodes within the intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'TyphoidVaccine',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Mode': {
            'default': 'Shedding',
            'description': 'The mode of contact transmission of typhoid targeted by the intervention.',
            'enum': ['Shedding', 'Dose', 'Exposures'],
            'type': 'enum',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Sim_Types': ['TYPHOID_SIM'],
        'class': 'TyphoidVaccine',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'TyphoidVaccine')

    def __init__(self, Changing_Effect=None, Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Effect=1, Intervention_Name='TyphoidVaccine', Mode=TyphoidVaccine_Mode_Enum.Shedding, New_Property_Value='', Sim_Types=['TYPHOID_SIM'], iv_type='IndividualTargeted', **kwargs):
        super(TyphoidVaccine, self).__init__(**kwargs)
        self.Changing_Effect = Changing_Effect
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Effect = Effect
        self.Intervention_Name = Intervention_Name
        self.Mode = (Mode.name if isinstance(Mode, Enum) else Mode)
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class UsageDependentBednet(BaseCampaign):
    _definition = {
        'Blocking_Config': {
            'description': 'Configures the rate of blocking for indoor mosquito feeds on individuals with an ITN; decays over time.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 3.75,
            'description': 'Unit cost per bednet (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Discard_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The event that is broadcast when an individual discards their bed net, either by replacing an existing net or due to the expiration timer.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of IndividualProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Dont_Allow_Duplicates': {
            'default': 0,
            'description': "If an individual's container has an intervention, set to true (1) to prevent them from receiving another copy of the intervention. Supported by all intervention classes.",
            'type': 'bool',
        },
        'Expiration_Period_Constant': {
            'default': 6,
            'depends-on': {
                'Expiration_Period_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Expiration_Period_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'The distribution type to use for assigning the expiration period to a usage-dependent bednet.',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Expiration_Period_Exponential': {
            'default': 6,
            'depends-on': {
                'Expiration_Period_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Expiration_Period_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Expiration_Period_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Expiration_Period_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Expiration_Period_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Expiration_Period_Kappa': {
            'default': 1,
            'depends-on': {
                'Expiration_Period_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Expiration_Period_Lambda': {
            'default': 1,
            'depends-on': {
                'Expiration_Period_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Expiration_Period_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Expiration_Period_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Expiration_Period_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Expiration_Period_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Expiration_Period_Max': {
            'default': 1,
            'depends-on': {
                'Expiration_Period_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Expiration_Period_Mean_1': {
            'default': 1,
            'depends-on': {
                'Expiration_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Expiration_Period_Mean_2': {
            'default': 1,
            'depends-on': {
                'Expiration_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Expiration_Period_Min': {
            'default': 0,
            'depends-on': {
                'Expiration_Period_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Expiration_Period_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Expiration_Period_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Expiration_Period_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Expiration_Period_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Expiration_Period_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Expiration_Period_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Expiration_Period_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Expiration_Period_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'UsageDependentBednet',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of the rate at which mosquitoes die, conditional on a successfully blocked feed; decays over time.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional IndividualProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Received_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'This parameter broadcasts when a new net is received, either the first net or a replacement net.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Sim_Types': ['MALARIA_SIM', 'VECTOR_SIM'],
        'Usage_Config_List': {
            'description': 'The list of WaningEffects whose effects are multiplied together to get the usage effect.',
            'type': 'list',
            'item_type': {
                'type': 'object',
            },
            'default': [],
        },
        'Using_Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'This parameter broadcasts each time step in which a bed net is used.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'class': 'UsageDependentBednet',
        'iv_type': 'IndividualTargeted',
    }
    _validator = ClassValidator(_definition, 'UsageDependentBednet')

    def __init__(self, Blocking_Config=None, Cost_To_Consumer=3.75, Discard_Event='', Disqualifying_Properties=[], Dont_Allow_Duplicates=False, Expiration_Period_Constant=6, Expiration_Period_Distribution=UsageDependentBednet_Expiration_Period_Distribution_Enum.CONSTANT_DISTRIBUTION, Expiration_Period_Exponential=6, Expiration_Period_Gaussian_Mean=6, Expiration_Period_Gaussian_Std_Dev=1, Expiration_Period_Kappa=1, Expiration_Period_Lambda=1, Expiration_Period_Log_Normal_Mu=6, Expiration_Period_Log_Normal_Sigma=1, Expiration_Period_Max=1, Expiration_Period_Mean_1=1, Expiration_Period_Mean_2=1, Expiration_Period_Min=0, Expiration_Period_Peak_2_Value=1, Expiration_Period_Poisson_Mean=6, Expiration_Period_Proportion_0=1, Expiration_Period_Proportion_1=1, Intervention_Name='UsageDependentBednet', Killing_Config=None, New_Property_Value='', Received_Event='', Sim_Types=['MALARIA_SIM', 'VECTOR_SIM'], Usage_Config_List=[], Using_Event='', iv_type='IndividualTargeted', **kwargs):
        super(UsageDependentBednet, self).__init__(**kwargs)
        self.Blocking_Config = Blocking_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Discard_Event = Discard_Event
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Dont_Allow_Duplicates = Dont_Allow_Duplicates
        self.Expiration_Period_Constant = Expiration_Period_Constant
        self.Expiration_Period_Distribution = (Expiration_Period_Distribution.name if isinstance(Expiration_Period_Distribution, Enum) else Expiration_Period_Distribution)
        self.Expiration_Period_Exponential = Expiration_Period_Exponential
        self.Expiration_Period_Gaussian_Mean = Expiration_Period_Gaussian_Mean
        self.Expiration_Period_Gaussian_Std_Dev = Expiration_Period_Gaussian_Std_Dev
        self.Expiration_Period_Kappa = Expiration_Period_Kappa
        self.Expiration_Period_Lambda = Expiration_Period_Lambda
        self.Expiration_Period_Log_Normal_Mu = Expiration_Period_Log_Normal_Mu
        self.Expiration_Period_Log_Normal_Sigma = Expiration_Period_Log_Normal_Sigma
        self.Expiration_Period_Max = Expiration_Period_Max
        self.Expiration_Period_Mean_1 = Expiration_Period_Mean_1
        self.Expiration_Period_Mean_2 = Expiration_Period_Mean_2
        self.Expiration_Period_Min = Expiration_Period_Min
        self.Expiration_Period_Peak_2_Value = Expiration_Period_Peak_2_Value
        self.Expiration_Period_Poisson_Mean = Expiration_Period_Poisson_Mean
        self.Expiration_Period_Proportion_0 = Expiration_Period_Proportion_0
        self.Expiration_Period_Proportion_1 = Expiration_Period_Proportion_1
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Received_Event = Received_Event
        self.Sim_Types = Sim_Types
        self.Usage_Config_List = Usage_Config_List
        self.Using_Event = Using_Event
        self.iv_type = iv_type



class AnimalFeedKill(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vector control (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'AnimalFeedKill',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of killing efficacy of the targeted stage. Use a waning effect class to specify how this effect decays over time.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'],
        'class': 'AnimalFeedKill',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'AnimalFeedKill')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Intervention_Name='AnimalFeedKill', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'], iv_type='NodeTargeted', **kwargs):
        super(AnimalFeedKill, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class ArtificialDiet(BaseCampaign):
    _definition = {
        'Artificial_Diet_Target': {
            'default': 'AD_WithinVillage',
            'description': 'The targeted deployment of artificial diet.',
            'enum': ['AD_WithinVillage', 'AD_OutsideVillage'],
            'type': 'enum',
        },
        'Attraction_Config': {
            'description': 'The fraction of vector feeds attracted to the artificial diet.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vector control (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'ArtificialDiet',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'],
        'class': 'ArtificialDiet',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'ArtificialDiet')

    def __init__(self, Artificial_Diet_Target=ArtificialDiet_Artificial_Diet_Target_Enum.AD_WithinVillage, Attraction_Config=None, Cost_To_Consumer=10, Disqualifying_Properties=[], Intervention_Name='ArtificialDiet', New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'], iv_type='NodeTargeted', **kwargs):
        super(ArtificialDiet, self).__init__(**kwargs)
        self.Artificial_Diet_Target = (Artificial_Diet_Target.name if isinstance(Artificial_Diet_Target, Enum) else Artificial_Diet_Target)
        self.Attraction_Config = Attraction_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class BirthTriggeredIV(BaseCampaign):
    _definition = {
        'Actual_IndividualIntervention_Config': {
            'description': 'The configuration of an actual individual intervention sought.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Demographic_Coverage': {
            'default': 1,
            'description': 'The fraction of individuals in the target demographic that will receive this intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Duration': {
            'default': -1,
            'description': 'The number of days to continue this intervention.',
            'max': 3.40282e+38,
            'min': -1,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'BirthTriggeredIV',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Sim_Types': ['*'],
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'class': 'BirthTriggeredIV',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'BirthTriggeredIV')

    def __init__(self, Actual_IndividualIntervention_Config=None, Demographic_Coverage=1, Disqualifying_Properties=[], Duration=-1, Intervention_Name='BirthTriggeredIV', New_Property_Value='', Property_Restrictions=[], Property_Restrictions_Within_Node=[], Sim_Types=['*'], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=BirthTriggeredIV_Target_Demographic_Enum.Everyone, Target_Gender=BirthTriggeredIV_Target_Gender_Enum.All, Target_Residents_Only=False, iv_type='NodeTargeted', **kwargs):
        super(BirthTriggeredIV, self).__init__(**kwargs)
        self.Actual_IndividualIntervention_Config = Actual_IndividualIntervention_Config
        self.Demographic_Coverage = Demographic_Coverage
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Duration = Duration
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Sim_Types = Sim_Types
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.iv_type = iv_type



class BroadcastNodeEvent(BaseCampaign):
    _definition = {
        'Broadcast_Event': {
            'Built-in': ['SheddingComplete'],
            'default': '',
            'description': 'The name of the node event to broadcast. This event must be set in the **Custom_Node_Events** configuration parameter.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Node_Events.*' or Built-in",
        },
        'Cost_To_Consumer': {
            'default': 0,
            'description': 'The unit cost of the intervention campaign that will be assigned to the specified nodes.',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['*'],
        'class': 'BroadcastNodeEvent',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'BroadcastNodeEvent')

    def __init__(self, Broadcast_Event='', Cost_To_Consumer=0, Sim_Types=['*'], iv_type='NodeTargeted', **kwargs):
        super(BroadcastNodeEvent, self).__init__(**kwargs)
        self.Broadcast_Event = Broadcast_Event
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class EnvironmentalDiagnostic(BaseCampaign):
    _definition = {
        'Base_Sensitivity': {
            'default': 1,
            'description': 'The likelihood that a positive measurement was made. If the contagion measurement is greater than the Sample_Threshold, a random number is drawn to determine if the detection was actually made.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Base_Specificity': {
            'default': 1,
            'description': 'The specificity of the diagnostic. This sets the proportion of the time that individuals without the condition being tested receive a negative diagnostic test. When set to 1, the diagnostic always accurately reflects the lack of having the condition. When set to zero, then individuals who do not have the condition always receive a false-positive diagnostic test.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Environment_IP_Key_Value': {
            'default': '',
            'description': 'An IndividualProperty key:value pair that indicates a specific transmission pool, typically used to identify a location.  If none is provided, the sample will be taken on the entire node.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Intervention_Name': {
            'default': 'EnvironmentalDiagnostic',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Negative_Diagnostic_Event': {
            'Built-in': ['SheddingComplete'],
            'default': '',
            'description': 'The event that will be broadcast to the node when the sample value is less than the threshold (e.g. the test is negative). If this is an empty string or set to NoTrigger, no event will be broadcast.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Node_Events.*' or Built-in",
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Positive_Diagnostic_Event': {
            'Built-in': ['SheddingComplete'],
            'default': '',
            'description': 'The event that will be broadcast to the node when the sample value is greater than the threshold (e.g. the test is positive). This cannot be an empty string or set to NoTrigger.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Node_Events.*' or Built-in",
        },
        'Sample_Threshold': {
            'default': 0,
            'description': 'The threshold that delineates a positive versus negative sampling result. If the sample is greater than the threshold, a positive finding will result; if the value is less than the threshold, it will be negative. This does not include values equal to the threshold so that the threshold can be set to zero; if the threshold is zero, the test is simply looking for any deposit in the transmission pool.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['ENVIRONMENTAL_SIM', 'TYPHOID_SIM'],
        'class': 'EnvironmentalDiagnostic',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'EnvironmentalDiagnostic')

    def __init__(self, Base_Sensitivity=1, Base_Specificity=1, Disqualifying_Properties=[], Environment_IP_Key_Value='', Intervention_Name='EnvironmentalDiagnostic', Negative_Diagnostic_Event='', New_Property_Value='', Positive_Diagnostic_Event='', Sample_Threshold=0, Sim_Types=['ENVIRONMENTAL_SIM', 'TYPHOID_SIM'], iv_type='NodeTargeted', **kwargs):
        super(EnvironmentalDiagnostic, self).__init__(**kwargs)
        self.Base_Sensitivity = Base_Sensitivity
        self.Base_Specificity = Base_Specificity
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Environment_IP_Key_Value = Environment_IP_Key_Value
        self.Intervention_Name = Intervention_Name
        self.Negative_Diagnostic_Event = Negative_Diagnostic_Event
        self.New_Property_Value = New_Property_Value
        self.Positive_Diagnostic_Event = Positive_Diagnostic_Event
        self.Sample_Threshold = Sample_Threshold
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class ImportPressure(BaseCampaign):
    _definition = {
        'Antigen': {
            'default': 0,
            'description': 'The antigenic base strain ID of the outbreak infection.',
            'max': 10,
            'min': 0,
            'type': 'integer',
        },
        'Daily_Import_Pressures': {
            'ascending': 0,
            'default': [],
            'description': 'The rate of per-day importation for each node that the intervention is distributed to.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'Vector Float',
        },
        'Durations': {
            'ascending': 0,
            'default': [],
            'description': 'The durations over which to apply import pressure.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'Vector Int',
        },
        'Genome': {
            'default': 0,
            'description': 'The genetic substrain ID of the outbreak infection.',
            'max': 16777200.0,
            'min': -1,
            'type': 'integer',
        },
        'Import_Age': {
            'default': 365,
            'description': 'The age (in days) of infected import cases.',
            'max': 43800,
            'min': 0,
            'type': 'float',
        },
        'Incubation_Period_Override': {
            'default': -1,
            'description': 'The incubation period, in days, that infected individuals will go through before becoming infectious. This value overrides the incubation period set in the configuration file. Set to -1 to honor the configuration parameter settings.',
            'max': 2147480000.0,
            'min': -1,
            'type': 'integer',
        },
        'Number_Cases_Per_Node': {
            'default': 1,
            'description': 'The number of new cases of Outbreak to import (per node).',
            'max': 2147480000.0,
            'min': 0,
            'type': 'integer',
        },
        'Probability_Of_Infection': {
            'default': 1,
            'description': 'Probability that each individual is infected. 1.0 is legacy functionality. 0.0 adds susceptibles to the population.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['GENERIC_SIM'],
        'class': 'ImportPressure',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'ImportPressure')

    def __init__(self, Antigen=0, Daily_Import_Pressures=[], Durations=[], Genome=0, Import_Age=365, Incubation_Period_Override=-1, Number_Cases_Per_Node=1, Probability_Of_Infection=1, Sim_Types=['GENERIC_SIM'], iv_type='NodeTargeted', **kwargs):
        super(ImportPressure, self).__init__(**kwargs)
        self.Antigen = Antigen
        self.Daily_Import_Pressures = Daily_Import_Pressures
        self.Durations = Durations
        self.Genome = Genome
        self.Import_Age = Import_Age
        self.Incubation_Period_Override = Incubation_Period_Override
        self.Number_Cases_Per_Node = Number_Cases_Per_Node
        self.Probability_Of_Infection = Probability_Of_Infection
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class InputEIR(BaseCampaign):
    _definition = {
        'Age_Dependence': {
            'default': 'OFF',
            'description': 'Determines how InputEIR depends on the age of the target.',
            'enum': ['OFF', 'LINEAR', 'SURFACE_AREA_DEPENDENT'],
            'type': 'enum',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'InputEIR',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Monthly_EIR': {
            'ascending': 0,
            'default': [],
            'description': 'An array of 12 elements that contain an entomological inoculation rate (EIR) for each month. Each value should be between 0 and 1000.',
            'max': 1000,
            'min': 0,
            'type': 'Vector Float',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['MALARIA_SIM'],
        'class': 'InputEIR',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'InputEIR')

    def __init__(self, Age_Dependence=InputEIR_Age_Dependence_Enum.OFF, Disqualifying_Properties=[], Intervention_Name='InputEIR', Monthly_EIR=[], New_Property_Value='', Sim_Types=['MALARIA_SIM'], iv_type='NodeTargeted', **kwargs):
        super(InputEIR, self).__init__(**kwargs)
        self.Age_Dependence = (Age_Dependence.name if isinstance(Age_Dependence, Enum) else Age_Dependence)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.Monthly_EIR = Monthly_EIR
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class InsectKillingFence(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vector control (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'InsectKillingFence',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration for the effects of killing of the targeted stage. Use a waning effect class to specify how this effect decays over time.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'],
        'class': 'InsectKillingFence',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'InsectKillingFence')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Intervention_Name='InsectKillingFence', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'], iv_type='NodeTargeted', **kwargs):
        super(InsectKillingFence, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class Larvicides(BaseCampaign):
    _definition = {
        'Blocking_Config': {
            'description': 'The configuration of larval habitat reduction and waning for targeted stage.',
            'type': 'idmType:WaningEffect',
        },
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vector control (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Habitat_Target': {
            'default': 'TEMPORARY_RAINFALL',
            'description': 'The larval habitat type targeted for larvicide.',
            'enum': ['TEMPORARY_RAINFALL', 'WATER_VEGETATION', 'HUMAN_POPULATION', 'CONSTANT', 'BRACKISH_SWAMP', 'MARSHY_STREAM', 'LINEAR_SPLINE', 'ALL_HABITATS'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'Larvicides',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of larval killing efficacy and waning for targeted stage.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'],
        'class': 'Larvicides',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'Larvicides')

    def __init__(self, Blocking_Config=None, Cost_To_Consumer=10, Disqualifying_Properties=[], Habitat_Target=Larvicides_Habitat_Target_Enum.TEMPORARY_RAINFALL, Intervention_Name='Larvicides', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'], iv_type='NodeTargeted', **kwargs):
        super(Larvicides, self).__init__(**kwargs)
        self.Blocking_Config = Blocking_Config
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Habitat_Target = (Habitat_Target.name if isinstance(Habitat_Target, Enum) else Habitat_Target)
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class MalariaChallenge(BaseCampaign):
    _definition = {
        'Challenge_Type': {
            'default': 'InfectiousBites',
            'description': 'The type of malaria challenge.',
            'enum': ['InfectiousBites', 'Sporozoites'],
            'type': 'enum',
        },
        'Coverage': {
            'default': 1,
            'description': 'The fraction of individuals receiving an intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Infectious_Bite_Count': {
            'default': 1,
            'description': 'The number of infectious bites.',
            'max': 1000,
            'min': 0,
            'type': 'integer',
        },
        'Intervention_Name': {
            'default': 'MalariaChallenge',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['MALARIA_SIM'],
        'Sporozoite_Count': {
            'default': 1,
            'description': 'The number of sporozoites.',
            'max': 1000,
            'min': 0,
            'type': 'integer',
        },
        'class': 'MalariaChallenge',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'MalariaChallenge')

    def __init__(self, Challenge_Type=MalariaChallenge_Challenge_Type_Enum.InfectiousBites, Coverage=1, Disqualifying_Properties=[], Infectious_Bite_Count=1, Intervention_Name='MalariaChallenge', New_Property_Value='', Sim_Types=['MALARIA_SIM'], Sporozoite_Count=1, iv_type='NodeTargeted', **kwargs):
        super(MalariaChallenge, self).__init__(**kwargs)
        self.Challenge_Type = (Challenge_Type.name if isinstance(Challenge_Type, Enum) else Challenge_Type)
        self.Coverage = Coverage
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Infectious_Bite_Count = Infectious_Bite_Count
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Sporozoite_Count = Sporozoite_Count
        self.iv_type = iv_type



class MigrateFamily(BaseCampaign):
    _definition = {
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Duration_At_Node_Constant': {
            'default': 6,
            'depends-on': {
                'Duration_At_Node_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'The distribution type to use for assigning the duration of time an individual or family spends at a destination node after intervention-based migration.',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Duration_At_Node_Exponential': {
            'default': 6,
            'depends-on': {
                'Duration_At_Node_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Duration_At_Node_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_At_Node_Kappa': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_At_Node_Lambda': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_At_Node_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Duration_At_Node_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Duration_At_Node_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Duration_At_Node_Max': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Mean_1': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_At_Node_Mean_2': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_At_Node_Min': {
            'default': 0,
            'depends-on': {
                'Duration_At_Node_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Duration_At_Node_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Duration_At_Node_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Duration_At_Node_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Constant': {
            'default': 6,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to all individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Distribution': {
            'default': 'CONSTANT_DISTRIBUTION',
            'description': 'The distribution type to use for assigning the duration of time an individual or family waits before migrating to the a destination node after intervention-based migration.',
            'enum': ['CONSTANT_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'GAUSSIAN_DISTRIBUTION', 'EXPONENTIAL_DISTRIBUTION', 'POISSON_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'DUAL_CONSTANT_DISTRIBUTION', 'WEIBULL_DISTRIBUTION', 'DUAL_EXPONENTIAL_DISTRIBUTION'],
            'type': 'enum',
        },
        'Duration_Before_Leaving_Exponential': {
            'default': 6,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean for an exponential distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Gaussian_Mean': {
            'default': 6,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The mean for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Gaussian_Std_Dev': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'GAUSSIAN_DISTRIBUTION',
            },
            'description': 'The standard deviation for a Gaussian distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Kappa': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The shape value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Lambda': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'WEIBULL_DISTRIBUTION',
            },
            'description': 'The scale value in a Weibull distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Log_Normal_Mu': {
            'default': 6,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The mean for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Log_Normal_Sigma': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'LOG_NORMAL_DISTRIBUTION',
            },
            'description': 'The width for a log-normal distribution.',
            'max': 3.40282e+38,
            'min': -3.40282e+38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Max': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The maximum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Mean_1': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the first exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Mean_2': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The mean of the second exponential distribution.',
            'max': 3.40282e+38,
            'min': 1.17549e-38,
            'type': 'float',
        },
        'Duration_Before_Leaving_Min': {
            'default': 0,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'UNIFORM_DISTRIBUTION',
            },
            'description': 'The minimum of the uniform distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Peak_2_Value': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The value to assign to the remaining individuals.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Poisson_Mean': {
            'default': 6,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'POISSON_DISTRIBUTION',
            },
            'description': 'The mean for a Poisson distribution.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Proportion_0': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'DUAL_CONSTANT_DISTRIBUTION',
            },
            'description': 'The proportion of individuals to assign a value of zero.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Duration_Before_Leaving_Proportion_1': {
            'default': 1,
            'depends-on': {
                'Duration_Before_Leaving_Distribution': 'DUAL_EXPONENTIAL_DISTRIBUTION',
            },
            'description': 'The proportion of individuals in the first exponential distribution.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'MigrateFamily',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Is_Moving': {
            'default': 0,
            'description': 'Set to true (1) to indicate the individual is permanently moving to a new home node for intervention-based migration.',
            'type': 'bool',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'NodeID_To_Migrate_To': {
            'default': 0,
            'description': 'The destination node ID for intervention-based migration.',
            'max': 4294970000.0,
            'min': 0,
            'type': 'integer',
        },
        'Sim_Types': ['*'],
        'class': 'MigrateFamily',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'MigrateFamily')

    def __init__(self, Disqualifying_Properties=[], Duration_At_Node_Constant=6, Duration_At_Node_Distribution=MigrateFamily_Duration_At_Node_Distribution_Enum.CONSTANT_DISTRIBUTION, Duration_At_Node_Exponential=6, Duration_At_Node_Gaussian_Mean=6, Duration_At_Node_Gaussian_Std_Dev=1, Duration_At_Node_Kappa=1, Duration_At_Node_Lambda=1, Duration_At_Node_Log_Normal_Mu=6, Duration_At_Node_Log_Normal_Sigma=1, Duration_At_Node_Max=1, Duration_At_Node_Mean_1=1, Duration_At_Node_Mean_2=1, Duration_At_Node_Min=0, Duration_At_Node_Peak_2_Value=1, Duration_At_Node_Poisson_Mean=6, Duration_At_Node_Proportion_0=1, Duration_At_Node_Proportion_1=1, Duration_Before_Leaving_Constant=6, Duration_Before_Leaving_Distribution=MigrateFamily_Duration_Before_Leaving_Distribution_Enum.CONSTANT_DISTRIBUTION, Duration_Before_Leaving_Exponential=6, Duration_Before_Leaving_Gaussian_Mean=6, Duration_Before_Leaving_Gaussian_Std_Dev=1, Duration_Before_Leaving_Kappa=1, Duration_Before_Leaving_Lambda=1, Duration_Before_Leaving_Log_Normal_Mu=6, Duration_Before_Leaving_Log_Normal_Sigma=1, Duration_Before_Leaving_Max=1, Duration_Before_Leaving_Mean_1=1, Duration_Before_Leaving_Mean_2=1, Duration_Before_Leaving_Min=0, Duration_Before_Leaving_Peak_2_Value=1, Duration_Before_Leaving_Poisson_Mean=6, Duration_Before_Leaving_Proportion_0=1, Duration_Before_Leaving_Proportion_1=1, Intervention_Name='MigrateFamily', Is_Moving=False, New_Property_Value='', NodeID_To_Migrate_To=0, Sim_Types=['*'], iv_type='NodeTargeted', **kwargs):
        super(MigrateFamily, self).__init__(**kwargs)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Duration_At_Node_Constant = Duration_At_Node_Constant
        self.Duration_At_Node_Distribution = (Duration_At_Node_Distribution.name if isinstance(Duration_At_Node_Distribution, Enum) else Duration_At_Node_Distribution)
        self.Duration_At_Node_Exponential = Duration_At_Node_Exponential
        self.Duration_At_Node_Gaussian_Mean = Duration_At_Node_Gaussian_Mean
        self.Duration_At_Node_Gaussian_Std_Dev = Duration_At_Node_Gaussian_Std_Dev
        self.Duration_At_Node_Kappa = Duration_At_Node_Kappa
        self.Duration_At_Node_Lambda = Duration_At_Node_Lambda
        self.Duration_At_Node_Log_Normal_Mu = Duration_At_Node_Log_Normal_Mu
        self.Duration_At_Node_Log_Normal_Sigma = Duration_At_Node_Log_Normal_Sigma
        self.Duration_At_Node_Max = Duration_At_Node_Max
        self.Duration_At_Node_Mean_1 = Duration_At_Node_Mean_1
        self.Duration_At_Node_Mean_2 = Duration_At_Node_Mean_2
        self.Duration_At_Node_Min = Duration_At_Node_Min
        self.Duration_At_Node_Peak_2_Value = Duration_At_Node_Peak_2_Value
        self.Duration_At_Node_Poisson_Mean = Duration_At_Node_Poisson_Mean
        self.Duration_At_Node_Proportion_0 = Duration_At_Node_Proportion_0
        self.Duration_At_Node_Proportion_1 = Duration_At_Node_Proportion_1
        self.Duration_Before_Leaving_Constant = Duration_Before_Leaving_Constant
        self.Duration_Before_Leaving_Distribution = (Duration_Before_Leaving_Distribution.name if isinstance(Duration_Before_Leaving_Distribution, Enum) else Duration_Before_Leaving_Distribution)
        self.Duration_Before_Leaving_Exponential = Duration_Before_Leaving_Exponential
        self.Duration_Before_Leaving_Gaussian_Mean = Duration_Before_Leaving_Gaussian_Mean
        self.Duration_Before_Leaving_Gaussian_Std_Dev = Duration_Before_Leaving_Gaussian_Std_Dev
        self.Duration_Before_Leaving_Kappa = Duration_Before_Leaving_Kappa
        self.Duration_Before_Leaving_Lambda = Duration_Before_Leaving_Lambda
        self.Duration_Before_Leaving_Log_Normal_Mu = Duration_Before_Leaving_Log_Normal_Mu
        self.Duration_Before_Leaving_Log_Normal_Sigma = Duration_Before_Leaving_Log_Normal_Sigma
        self.Duration_Before_Leaving_Max = Duration_Before_Leaving_Max
        self.Duration_Before_Leaving_Mean_1 = Duration_Before_Leaving_Mean_1
        self.Duration_Before_Leaving_Mean_2 = Duration_Before_Leaving_Mean_2
        self.Duration_Before_Leaving_Min = Duration_Before_Leaving_Min
        self.Duration_Before_Leaving_Peak_2_Value = Duration_Before_Leaving_Peak_2_Value
        self.Duration_Before_Leaving_Poisson_Mean = Duration_Before_Leaving_Poisson_Mean
        self.Duration_Before_Leaving_Proportion_0 = Duration_Before_Leaving_Proportion_0
        self.Duration_Before_Leaving_Proportion_1 = Duration_Before_Leaving_Proportion_1
        self.Intervention_Name = Intervention_Name
        self.Is_Moving = Is_Moving
        self.New_Property_Value = New_Property_Value
        self.NodeID_To_Migrate_To = NodeID_To_Migrate_To
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class MosquitoRelease(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 0,
            'description': 'Cost of each mosquito release.',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'MosquitoRelease',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Mated_Genetics': {
            'depends-on': {
                'Released_Gender': 'VECTOR_FEMALE',
            },
            'description': 'The genetic properties of the mate if released mosquitoes have mated, e.g. HEG and pesticide resistance.',
            'HEG': {
                'default': 'WILD',
                'description': 'HEG characteristics of released mosquitoes.',
                'enum': ['WILD', 'HALF', 'FULL', 'NotMated'],
                'type': 'enum',
            },
            'Pesticide_Resistance': {
                'default': 'WILD',
                'description': 'The pesticide resistance characteristics of released mosquitoes.',
                'enum': ['WILD', 'HALF', 'FULL', 'NotMated'],
                'type': 'enum',
            },
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Released_Gender': {
            'default': 'VECTOR_FEMALE',
            'description': 'The gender of released mosquitoes.',
            'enum': ['VECTOR_FEMALE', 'VECTOR_MALE'],
            'type': 'enum',
        },
        'Released_Genetics': {
            'description': 'The genetic properties of the released mosquito, e.g. HEG and pesticide resistance.',
            'HEG': {
                'default': 'WILD',
                'description': 'HEG characteristics of released mosquitoes.',
                'enum': ['WILD', 'HALF', 'FULL', 'NotMated'],
                'type': 'enum',
            },
            'Pesticide_Resistance': {
                'default': 'WILD',
                'description': 'The pesticide resistance characteristics of released mosquitoes.',
                'enum': ['WILD', 'HALF', 'FULL', 'NotMated'],
                'type': 'enum',
            },
        },
        'Released_Number': {
            'default': 10000,
            'description': 'The number of mosquitoes released in the intervention.',
            'max': 100000000.0,
            'min': 1,
            'type': 'integer',
        },
        'Released_Species': {
            'default': 'UNINITIALIZED STRING',
            'description': 'The name of the released mosquito species.',
            'type': 'Constrained String',
            'value_source': '<configuration>:Vector_Species_Params.*',
        },
        'Released_Sterility': {
            'default': 'VECTOR_FERTILE',
            'description': 'The sterility of released mosquitoes.',
            'enum': ['VECTOR_FERTILE', 'VECTOR_STERILE'],
            'type': 'enum',
        },
        'Released_Wolbachia': {
            'default': 'WOLBACHIA_FREE',
            'description': 'The Wolbachia type of released mosquitoes.',
            'enum': ['WOLBACHIA_FREE', 'VECTOR_WOLBACHIA_A', 'VECTOR_WOLBACHIA_B', 'VECTOR_WOLBACHIA_AB'],
            'type': 'enum',
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM'],
        'class': 'MosquitoRelease',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'MosquitoRelease')

    def __init__(self, Cost_To_Consumer=0, Disqualifying_Properties=[], Intervention_Name='MosquitoRelease', Mated_Genetics=None, New_Property_Value='', Released_Gender=MosquitoRelease_Released_Gender_Enum.VECTOR_FEMALE, Released_Genetics=None, Released_Number=10000, Released_Species='UNINITIALIZED STRING', Released_Sterility=MosquitoRelease_Released_Sterility_Enum.VECTOR_FERTILE, Released_Wolbachia=MosquitoRelease_Released_Wolbachia_Enum.WOLBACHIA_FREE, Sim_Types=['VECTOR_SIM', 'MALARIA_SIM'], iv_type='NodeTargeted', **kwargs):
        super(MosquitoRelease, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.Mated_Genetics = Mated_Genetics
        self.New_Property_Value = New_Property_Value
        self.Released_Gender = (Released_Gender.name if isinstance(Released_Gender, Enum) else Released_Gender)
        self.Released_Genetics = Released_Genetics
        self.Released_Number = Released_Number
        self.Released_Species = Released_Species
        self.Released_Sterility = (Released_Sterility.name if isinstance(Released_Sterility, Enum) else Released_Sterility)
        self.Released_Wolbachia = (Released_Wolbachia.name if isinstance(Released_Wolbachia, Enum) else Released_Wolbachia)
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class MultiNodeInterventionDistributor(BaseCampaign):
    _definition = {
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'MultiNodeInterventionDistributor',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Node_Intervention_List': {
            'description': 'A list of nested JSON objects for the multi-node-level interventions to be distributed by this intervention.',
            'type': 'idmAbstractType:NodeIntervention',
        },
        'Sim_Types': ['*'],
        'class': 'MultiNodeInterventionDistributor',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'MultiNodeInterventionDistributor')

    def __init__(self, Disqualifying_Properties=[], Intervention_Name='MultiNodeInterventionDistributor', New_Property_Value='', Node_Intervention_List=None, Sim_Types=['*'], iv_type='NodeTargeted', **kwargs):
        super(MultiNodeInterventionDistributor, self).__init__(**kwargs)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Node_Intervention_List = Node_Intervention_List
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class NLHTIVNode(BaseCampaign):
    _definition = {
        'Actual_NodeIntervention_Config': {
            'description': 'The configuration of the actual node-level intervention sought.',
            'type': 'idmAbstractType:NodeIntervention',
        },
        'Blackout_Event_Trigger': {
            'Built-in': ['SheddingComplete'],
            'default': '',
            'description': 'The event to broadcast if an intervention cannot be distributed due to the Blackout_Period.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Node_Events.*' or Built-in",
        },
        'Blackout_On_First_Occurrence': {
            'default': 0,
            'description': 'If set to true (1), individuals will enter the blackout period after the first occurrence of an event in the Trigger_Condition_List.',
            'type': 'bool',
        },
        'Blackout_Period': {
            'default': 0,
            'description': 'After the initial intervention distribution, the time, in days, to wait before distributing the intervention again. If it cannot distribute due to the blackout period, it will broadcast the user-defined Blackout_Event_Trigger.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Demographic_Coverage': {
            'default': 1,
            'description': 'The fraction of individuals in the target demographic that will receive this intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Duration': {
            'default': -1,
            'description': 'The number of days to continue this intervention.',
            'max': 3.40282e+38,
            'min': -1,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'NLHTIVNode',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Sim_Types': ['*'],
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Trigger_Condition_List': {
            'Built-in': ['SheddingComplete'],
            'default': [],
            'description': 'A list of events that trigger a health seeking intervention.',
            'type': 'Vector Constrained String',
            'value_source': "'<configuration>:Custom_Node_Events.*' or Built-in",
        },
        'class': 'NLHTIVNode',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'NLHTIVNode')

    def __init__(self, Actual_NodeIntervention_Config=None, Blackout_Event_Trigger='', Blackout_On_First_Occurrence=False, Blackout_Period=0, Demographic_Coverage=1, Disqualifying_Properties=[], Duration=-1, Intervention_Name='NLHTIVNode', New_Property_Value='', Node_Property_Restrictions=[], Property_Restrictions=[], Property_Restrictions_Within_Node=[], Sim_Types=['*'], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=NLHTIVNode_Target_Demographic_Enum.Everyone, Target_Gender=NLHTIVNode_Target_Gender_Enum.All, Target_Residents_Only=False, Trigger_Condition_List=[], iv_type='NodeTargeted', **kwargs):
        super(NLHTIVNode, self).__init__(**kwargs)
        self.Actual_NodeIntervention_Config = Actual_NodeIntervention_Config
        self.Blackout_Event_Trigger = Blackout_Event_Trigger
        self.Blackout_On_First_Occurrence = Blackout_On_First_Occurrence
        self.Blackout_Period = Blackout_Period
        self.Demographic_Coverage = Demographic_Coverage
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Duration = Duration
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Sim_Types = Sim_Types
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Trigger_Condition_List = Trigger_Condition_List
        self.iv_type = iv_type



class NodeLevelHealthTriggeredIV(BaseCampaign):
    _definition = {
        'Actual_IndividualIntervention_Config': {
            'description': 'The configuration of an actual individual intervention sought.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Actual_NodeIntervention_Config': {
            'description': 'The configuration of the actual node-level intervention sought.',
            'type': 'idmAbstractType:NodeIntervention',
        },
        'Blackout_Event_Trigger': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The event to broadcast if an intervention cannot be distributed due to the Blackout_Period.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Blackout_On_First_Occurrence': {
            'default': 0,
            'description': 'If set to true (1), individuals will enter the blackout period after the first occurrence of an event in the Trigger_Condition_List.',
            'type': 'bool',
        },
        'Blackout_Period': {
            'default': 0,
            'description': 'After the initial intervention distribution, the time, in days, to wait before distributing the intervention again. If it cannot distribute due to the blackout period, it will broadcast the user-defined Blackout_Event_Trigger.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Demographic_Coverage': {
            'default': 1,
            'description': 'The fraction of individuals in the target demographic that will receive this intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Distribute_On_Return_Home': {
            'default': 0,
            'description': 'When set to true (1), individuals will receive an intervention upon returning home if that intervention was originally distributed while the individual was away.',
            'type': 'bool',
        },
        'Duration': {
            'default': -1,
            'description': 'The number of days to continue this intervention.',
            'max': 3.40282e+38,
            'min': -1,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'NodeLevelHealthTriggeredIV',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Sim_Types': ['*'],
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Trigger_Condition_List': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': [],
            'description': 'A list of events that trigger a health seeking intervention.',
            'type': 'Vector Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'class': 'NodeLevelHealthTriggeredIV',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'NodeLevelHealthTriggeredIV')

    def __init__(self, Actual_IndividualIntervention_Config=None, Actual_NodeIntervention_Config=None, Blackout_Event_Trigger='', Blackout_On_First_Occurrence=False, Blackout_Period=0, Demographic_Coverage=1, Disqualifying_Properties=[], Distribute_On_Return_Home=False, Duration=-1, Intervention_Name='NodeLevelHealthTriggeredIV', New_Property_Value='', Node_Property_Restrictions=[], Property_Restrictions=[], Property_Restrictions_Within_Node=[], Sim_Types=['*'], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=NodeLevelHealthTriggeredIV_Target_Demographic_Enum.Everyone, Target_Gender=NodeLevelHealthTriggeredIV_Target_Gender_Enum.All, Target_Residents_Only=False, Trigger_Condition_List=[], iv_type='NodeTargeted', **kwargs):
        super(NodeLevelHealthTriggeredIV, self).__init__(**kwargs)
        self.Actual_IndividualIntervention_Config = Actual_IndividualIntervention_Config
        self.Actual_NodeIntervention_Config = Actual_NodeIntervention_Config
        self.Blackout_Event_Trigger = Blackout_Event_Trigger
        self.Blackout_On_First_Occurrence = Blackout_On_First_Occurrence
        self.Blackout_Period = Blackout_Period
        self.Demographic_Coverage = Demographic_Coverage
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Distribute_On_Return_Home = Distribute_On_Return_Home
        self.Duration = Duration
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Sim_Types = Sim_Types
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Trigger_Condition_List = Trigger_Condition_List
        self.iv_type = iv_type



class NodeLevelHealthTriggeredIVScaleUpSwitch(BaseCampaign):
    _definition = {
        'Actual_IndividualIntervention_Config': {
            'description': 'The configuration of an actual individual intervention sought.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Actual_NodeIntervention_Config': {
            'description': 'The configuration of the actual node-level intervention sought.',
            'type': 'idmAbstractType:NodeIntervention',
        },
        'Blackout_Event_Trigger': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The event to broadcast if an intervention cannot be distributed due to the Blackout_Period.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'Blackout_On_First_Occurrence': {
            'default': 0,
            'description': 'If set to true (1), individuals will enter the blackout period after the first occurrence of an event in the Trigger_Condition_List.',
            'type': 'bool',
        },
        'Blackout_Period': {
            'default': 0,
            'description': 'After the initial intervention distribution, the time, in days, to wait before distributing the intervention again. If it cannot distribute due to the blackout period, it will broadcast the user-defined Blackout_Event_Trigger.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Demographic_Coverage': {
            'default': 1,
            'description': 'The fraction of individuals in the target demographic that will receive this intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Demographic_Coverage_Time_Profile': {
            'default': 'Immediate',
            'description': 'Profile of ramp up to demographic coverage.',
            'enum': ['Immediate', 'Linear', 'Sigmoid'],
            'type': 'enum',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Distribute_On_Return_Home': {
            'default': 0,
            'description': 'When set to true (1), individuals will receive an intervention upon returning home if that intervention was originally distributed while the individual was away.',
            'type': 'bool',
        },
        'Duration': {
            'default': -1,
            'description': 'The number of days to continue this intervention.',
            'max': 3.40282e+38,
            'min': -1,
            'type': 'float',
        },
        'Initial_Demographic_Coverage': {
            'default': 0,
            'description': 'The initial level of demographic coverage if using linear scale up.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'NodeLevelHealthTriggeredIVScaleUpSwitch',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Not_Covered_IndividualIntervention_Configs': {
            'description': 'The array of interventions that is distributed if an individual qualifies according to all parameters except Demographic_Coverage.',
            'type': 'idmAbstractType:IndividualIntervention',
        },
        'Primary_Time_Constant': {
            'default': 1,
            'description': 'The time to full scale-up of demographic coverage.',
            'max': 365000,
            'min': 0,
            'type': 'float',
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Sim_Types': ['*'],
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Trigger_Condition_List': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': [],
            'description': 'A list of events that trigger a health seeking intervention.',
            'type': 'Vector Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'class': 'NodeLevelHealthTriggeredIVScaleUpSwitch',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'NodeLevelHealthTriggeredIVScaleUpSwitch')

    def __init__(self, Actual_IndividualIntervention_Config=None, Actual_NodeIntervention_Config=None, Blackout_Event_Trigger='', Blackout_On_First_Occurrence=False, Blackout_Period=0, Demographic_Coverage=1, Demographic_Coverage_Time_Profile=NodeLevelHealthTriggeredIVScaleUpSwitch_Demographic_Coverage_Time_Profile_Enum.Immediate, Disqualifying_Properties=[], Distribute_On_Return_Home=False, Duration=-1, Initial_Demographic_Coverage=0, Intervention_Name='NodeLevelHealthTriggeredIVScaleUpSwitch', New_Property_Value='', Node_Property_Restrictions=[], Not_Covered_IndividualIntervention_Configs=None, Primary_Time_Constant=1, Property_Restrictions=[], Property_Restrictions_Within_Node=[], Sim_Types=['*'], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=NodeLevelHealthTriggeredIVScaleUpSwitch_Target_Demographic_Enum.Everyone, Target_Gender=NodeLevelHealthTriggeredIVScaleUpSwitch_Target_Gender_Enum.All, Target_Residents_Only=False, Trigger_Condition_List=[], iv_type='NodeTargeted', **kwargs):
        super(NodeLevelHealthTriggeredIVScaleUpSwitch, self).__init__(**kwargs)
        self.Actual_IndividualIntervention_Config = Actual_IndividualIntervention_Config
        self.Actual_NodeIntervention_Config = Actual_NodeIntervention_Config
        self.Blackout_Event_Trigger = Blackout_Event_Trigger
        self.Blackout_On_First_Occurrence = Blackout_On_First_Occurrence
        self.Blackout_Period = Blackout_Period
        self.Demographic_Coverage = Demographic_Coverage
        self.Demographic_Coverage_Time_Profile = (Demographic_Coverage_Time_Profile.name if isinstance(Demographic_Coverage_Time_Profile, Enum) else Demographic_Coverage_Time_Profile)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Distribute_On_Return_Home = Distribute_On_Return_Home
        self.Duration = Duration
        self.Initial_Demographic_Coverage = Initial_Demographic_Coverage
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Not_Covered_IndividualIntervention_Configs = Not_Covered_IndividualIntervention_Configs
        self.Primary_Time_Constant = Primary_Time_Constant
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Sim_Types = Sim_Types
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Trigger_Condition_List = Trigger_Condition_List
        self.iv_type = iv_type



class NodePropertyValueChanger(BaseCampaign):
    _definition = {
        'Daily_Probability': {
            'default': 1,
            'description': 'The daily probability that an individual will move to the Target_Property_Value.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'NodePropertyValueChanger',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Maximum_Duration': {
            'default': 3.40282e+38,
            'description': 'The maximum amount of time individuals have to move to a new group. This timing works in conjunction with Daily_Probability.',
            'max': 3.40282e+38,
            'min': -1,
            'type': 'float',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Revert': {
            'default': 0,
            'description': 'The number of days before an individual moves back to their original group.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['*'],
        'Target_NP_Key_Value': {
            'default': '',
            'description': 'The NodeProperty key:value pair, as defined in the demographics file, to assign to the node.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'class': 'NodePropertyValueChanger',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'NodePropertyValueChanger')

    def __init__(self, Daily_Probability=1, Disqualifying_Properties=[], Intervention_Name='NodePropertyValueChanger', Maximum_Duration=3.40282e+38, New_Property_Value='', Revert=0, Sim_Types=['*'], Target_NP_Key_Value='', iv_type='NodeTargeted', **kwargs):
        super(NodePropertyValueChanger, self).__init__(**kwargs)
        self.Daily_Probability = Daily_Probability
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.Maximum_Duration = Maximum_Duration
        self.New_Property_Value = New_Property_Value
        self.Revert = Revert
        self.Sim_Types = Sim_Types
        self.Target_NP_Key_Value = Target_NP_Key_Value
        self.iv_type = iv_type



class Outbreak(BaseCampaign):
    _definition = {
        'Antigen': {
            'default': 0,
            'description': 'The antigenic base strain ID of the outbreak infection.',
            'max': 10,
            'min': 0,
            'type': 'integer',
        },
        'Genome': {
            'default': 0,
            'description': 'The genetic substrain ID of the outbreak infection.',
            'max': 16777200.0,
            'min': -1,
            'type': 'integer',
        },
        'Import_Age': {
            'default': 365,
            'description': 'The age (in days) of infected import cases.',
            'max': 43800,
            'min': 0,
            'type': 'float',
        },
        'Incubation_Period_Override': {
            'default': -1,
            'description': 'The incubation period, in days, that infected individuals will go through before becoming infectious. This value overrides the incubation period set in the configuration file. Set to -1 to honor the configuration parameter settings.',
            'max': 2147480000.0,
            'min': -1,
            'type': 'integer',
        },
        'Number_Cases_Per_Node': {
            'default': 1,
            'description': 'The number of new cases of Outbreak to import (per node).',
            'max': 2147480000.0,
            'min': 0,
            'type': 'integer',
        },
        'Probability_Of_Infection': {
            'default': 1,
            'description': 'Probability that each individual is infected. 1.0 is legacy functionality. 0.0 adds susceptibles to the population.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Sim_Types': ['GENERIC_SIM', 'VECTOR_SIM', 'MALARIA_SIM', 'AIRBORNE_SIM', 'POLIO_SIM', 'TBHIV_SIM', 'STI_SIM', 'HIV_SIM', 'PY_SIM', 'TYPHOID_SIM'],
        'class': 'Outbreak',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'Outbreak')

    def __init__(self, Antigen=0, Genome=0, Import_Age=365, Incubation_Period_Override=-1, Number_Cases_Per_Node=1, Probability_Of_Infection=1, Sim_Types=['GENERIC_SIM', 'VECTOR_SIM', 'MALARIA_SIM', 'AIRBORNE_SIM', 'POLIO_SIM', 'TBHIV_SIM', 'STI_SIM', 'HIV_SIM', 'PY_SIM', 'TYPHOID_SIM'], iv_type='NodeTargeted', **kwargs):
        super(Outbreak, self).__init__(**kwargs)
        self.Antigen = Antigen
        self.Genome = Genome
        self.Import_Age = Import_Age
        self.Incubation_Period_Override = Incubation_Period_Override
        self.Number_Cases_Per_Node = Number_Cases_Per_Node
        self.Probability_Of_Infection = Probability_Of_Infection
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class OutdoorRestKill(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vector control (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'OutdoorRestKill',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration for the effects of killing of the targeted stage. Use a waning effect class to specify how this effect decays over time.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'],
        'class': 'OutdoorRestKill',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'OutdoorRestKill')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Intervention_Name='OutdoorRestKill', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'], iv_type='NodeTargeted', **kwargs):
        super(OutdoorRestKill, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class OvipositionTrap(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vector control (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Habitat_Target': {
            'default': 'TEMPORARY_RAINFALL',
            'description': 'The oviposition habitat type targeted by oviposition traps.',
            'enum': ['TEMPORARY_RAINFALL', 'WATER_VEGETATION', 'HUMAN_POPULATION', 'CONSTANT', 'BRACKISH_SWAMP', 'MARSHY_STREAM', 'LINEAR_SPLINE', 'ALL_HABITATS'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'OvipositionTrap',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': "The configuration of the killing effects for the fraction of oviposition cycles that end in the female mosquito's death.  If there is skip oviposition, this is not the mortality per skip but the effective net mortality per gonotrophic cycle over all skips.",
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'],
        'class': 'OvipositionTrap',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'OvipositionTrap')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Habitat_Target=OvipositionTrap_Habitat_Target_Enum.TEMPORARY_RAINFALL, Intervention_Name='OvipositionTrap', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'], iv_type='NodeTargeted', **kwargs):
        super(OvipositionTrap, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Habitat_Target = (Habitat_Target.name if isinstance(Habitat_Target, Enum) else Habitat_Target)
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class PolioNodeSurvey(BaseCampaign):
    _definition = {
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'PolioNodeSurvey',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['POLIO_SIM'],
        'class': 'PolioNodeSurvey',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'PolioNodeSurvey')

    def __init__(self, Disqualifying_Properties=[], Intervention_Name='PolioNodeSurvey', New_Property_Value='', Sim_Types=['POLIO_SIM'], iv_type='NodeTargeted', **kwargs):
        super(PolioNodeSurvey, self).__init__(**kwargs)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class ScaleLarvalHabitat(BaseCampaign):
    _definition = {
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'ScaleLarvalHabitat',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Larval_Habitat_Multiplier': {
            'description': 'The value by which to scale the larval habitat availability specified in the configuration file with Larval_Habitat_Types across all habitat types, for specific habitat types, or for specific mosquito species within each habitat type.',
            'type': 'object',
            'subclasses': 'LarvalHabitatMultiplier',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'],
        'class': 'ScaleLarvalHabitat',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'ScaleLarvalHabitat')

    def __init__(self, Disqualifying_Properties=[], Intervention_Name='ScaleLarvalHabitat', Larval_Habitat_Multiplier=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'], iv_type='NodeTargeted', **kwargs):
        super(ScaleLarvalHabitat, self).__init__(**kwargs)
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.Larval_Habitat_Multiplier = Larval_Habitat_Multiplier
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class SpaceSpraying(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vector control (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Habitat_Target': {
            'default': 'TEMPORARY_RAINFALL',
            'description': 'The larval habitat type targeted for habitat reduction.',
            'enum': ['TEMPORARY_RAINFALL', 'WATER_VEGETATION', 'HUMAN_POPULATION', 'CONSTANT', 'BRACKISH_SWAMP', 'MARSHY_STREAM', 'LINEAR_SPLINE', 'ALL_HABITATS'],
            'type': 'enum',
        },
        'Intervention_Name': {
            'default': 'SpaceSpraying',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration of killing efficacy and waning for space spaying.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Reduction_Config': {
            'description': 'The configuration of larval habitat reduction and waning for space spraying.',
            'type': 'idmType:WaningEffect',
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'],
        'Spray_Kill_Target': {
            'default': 'SpaceSpray_FemalesOnly',
            'description': 'The gender kill-target of vector control interventions.',
            'enum': ['SpaceSpray_FemalesOnly', 'SpaceSpray_MalesOnly', 'SpaceSpray_FemalesAndMales', 'SpaceSpray_Indoor'],
            'type': 'enum',
        },
        'class': 'SpaceSpraying',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'SpaceSpraying')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Habitat_Target=SpaceSpraying_Habitat_Target_Enum.TEMPORARY_RAINFALL, Intervention_Name='SpaceSpraying', Killing_Config=None, New_Property_Value='', Reduction_Config=None, Sim_Types=['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'], Spray_Kill_Target=SpaceSpraying_Spray_Kill_Target_Enum.SpaceSpray_FemalesOnly, iv_type='NodeTargeted', **kwargs):
        super(SpaceSpraying, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Habitat_Target = (Habitat_Target.name if isinstance(Habitat_Target, Enum) else Habitat_Target)
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Reduction_Config = Reduction_Config
        self.Sim_Types = Sim_Types
        self.Spray_Kill_Target = (Spray_Kill_Target.name if isinstance(Spray_Kill_Target, Enum) else Spray_Kill_Target)
        self.iv_type = iv_type



class SpatialRepellent(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vector control (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'SpatialRepellent',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Repellency_Config': {
            'description': 'The configuration of efficacy and waning for spatial repellent.',
            'type': 'idmType:WaningEffect',
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'],
        'class': 'SpatialRepellent',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'SpatialRepellent')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Intervention_Name='SpatialRepellent', New_Property_Value='', Repellency_Config=None, Sim_Types=['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'], iv_type='NodeTargeted', **kwargs):
        super(SpatialRepellent, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.New_Property_Value = New_Property_Value
        self.Repellency_Config = Repellency_Config
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class SugarTrap(BaseCampaign):
    _definition = {
        'Cost_To_Consumer': {
            'default': 10,
            'description': 'Unit cost per vector control (unamortized).',
            'max': 999999,
            'min': 0,
            'type': 'float',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Intervention_Name': {
            'default': 'SugarTrap',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Killing_Config': {
            'description': 'The configuration for the effects of killing of the targeted stage. Use a waning effect class to specify how this effect decays over time.',
            'type': 'idmType:WaningEffect',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'],
        'class': 'SugarTrap',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'SugarTrap')

    def __init__(self, Cost_To_Consumer=10, Disqualifying_Properties=[], Intervention_Name='SugarTrap', Killing_Config=None, New_Property_Value='', Sim_Types=['VECTOR_SIM', 'MALARIA_SIM', 'DENGUE_SIM'], iv_type='NodeTargeted', **kwargs):
        super(SugarTrap, self).__init__(**kwargs)
        self.Cost_To_Consumer = Cost_To_Consumer
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Intervention_Name = Intervention_Name
        self.Killing_Config = Killing_Config
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.iv_type = iv_type



class TyphoidWASH(BaseCampaign):
    _definition = {
        'Changing_Effect': {
            'description': 'A highly configurable effect that changes over time.',
            'type': 'idmType:WaningEffect',
        },
        'Disqualifying_Properties': {
            'default': [],
            'description': 'A list of NodeProperty key:value pairs that cause an intervention to be aborted. Generally used to control the flow of health care access. For example, to prevent the same individual from accessing health care via two different routes at the same time.',
            'type': 'Dynamic String Set',
            'value_source': '',
        },
        'Effect': {
            'default': 1,
            'description': 'The efficacy of the Typhoid vaccine intervention. For example, a value of 1 would be 100 percent efficacy for all targeted nodes within the intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Intervention_Name': {
            'default': 'TyphoidWASH',
            'description': 'The optional name used to refer to this intervention as a means to differentiate it from others that use the same class.',
            'type': 'string',
        },
        'Mode': {
            'default': 'Shedding',
            'description': 'The mode of contact transmission of typhoid targeted by the intervention.',
            'enum': ['Shedding', 'Dose', 'Exposures'],
            'type': 'enum',
        },
        'New_Property_Value': {
            'default': '',
            'description': 'An optional NodeProperty key:value pair that will be assigned when the intervention is distributed. Generally used to indicate the broad category of health care cascade to which an intervention belongs to prevent individuals from accessing care through multiple pathways.',
            'type': 'Constrained String',
            'value_source': "'<demographics>::NodeProperties.*.Property':'<demographics>::NodeProperties.*.Values'",
        },
        'Sim_Types': ['TYPHOID_SIM'],
        'Targeted_Individual_Properties': {
            'default': 'default',
            'description': 'Individual Property key-value pairs to be targeted (optional).',
            'type': 'string',
        },
        'Use_Property_Targeting': {
            'default': 1,
            'description': 'Set to 1 (true) -- or omit -- if you want to use the Targeted_Individual_Property parameter to limit the effect of this intervention to just certain individuals. Set to 0 to apply effect to everyone.',
            'type': 'bool',
        },
        'class': 'TyphoidWASH',
        'iv_type': 'NodeTargeted',
    }
    _validator = ClassValidator(_definition, 'TyphoidWASH')

    def __init__(self, Changing_Effect=None, Disqualifying_Properties=[], Effect=1, Intervention_Name='TyphoidWASH', Mode=TyphoidWASH_Mode_Enum.Shedding, New_Property_Value='', Sim_Types=['TYPHOID_SIM'], Targeted_Individual_Properties='default', Use_Property_Targeting=True, iv_type='NodeTargeted', **kwargs):
        super(TyphoidWASH, self).__init__(**kwargs)
        self.Changing_Effect = Changing_Effect
        self.Disqualifying_Properties = Disqualifying_Properties
        self.Effect = Effect
        self.Intervention_Name = Intervention_Name
        self.Mode = (Mode.name if isinstance(Mode, Enum) else Mode)
        self.New_Property_Value = New_Property_Value
        self.Sim_Types = Sim_Types
        self.Targeted_Individual_Properties = Targeted_Individual_Properties
        self.Use_Property_Targeting = Use_Property_Targeting
        self.iv_type = iv_type



class NodeSetAll(BaseCampaign):
    _definition = {
        'class': 'NodeSetAll',
    }
    _validator = ClassValidator(_definition, 'NodeSetAll')

    def __init__(self, **kwargs):
        super(NodeSetAll, self).__init__(**kwargs)



class NodeSetNodeList(BaseCampaign):
    _definition = {
        'Node_List': {
            'description': 'A comma-separated list of node IDs in which this event will occur.',
            'type': 'list',
            'item_type': {
                'description': 'Id of Node',
                'min': 0,
                'type': 'integer',
            },
            'default': [],
        },
        'class': 'NodeSetNodeList',
    }
    _validator = ClassValidator(_definition, 'NodeSetNodeList')

    def __init__(self, Node_List=[], **kwargs):
        super(NodeSetNodeList, self).__init__(**kwargs)
        self.Node_List = Node_List



class NodeSetPolygon(BaseCampaign):
    _definition = {
        'Polygon_Format': {
            'default': 'SHAPE',
            'description': 'The type of polygon to create.',
            'enum': ['SHAPE'],
            'type': 'enum',
        },
        'Vertices': {
            'default': 'UNINITIALIZED STRING',
            'description': 'Comma-separated list of latitude/longitude points that bound a polygon.',
            'type': 'string',
        },
        'class': 'NodeSetPolygon',
    }
    _validator = ClassValidator(_definition, 'NodeSetPolygon')

    def __init__(self, Polygon_Format=NodeSetPolygon_Polygon_Format_Enum.SHAPE, Vertices='UNINITIALIZED STRING', **kwargs):
        super(NodeSetPolygon, self).__init__(**kwargs)
        self.Polygon_Format = (Polygon_Format.name if isinstance(Polygon_Format, Enum) else Polygon_Format)
        self.Vertices = Vertices



class Action(BaseCampaign):
    _definition = {
        'Event_To_Broadcast': {
            'default': 'UNINITIALIZED STRING',
            'description': 'The action event to occur when the specified trigger value in the Threshold parameter is met. At least one action must be defined for Event_To_Broadcast.',
            'type': 'string',
        },
        'Event_Type': {
            'default': 'INDIVIDUAL',
            'description': 'The type of event to be broadcast.',
            'enum': ['INDIVIDUAL', 'NODE', 'COORDINATOR'],
            'type': 'enum',
        },
        'Threshold': {
            'default': 0,
            'description': 'The threshold value that triggers the specified action for the Event_To_Broadcast parameter.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'class': 'Action',
    }
    _validator = ClassValidator(_definition, 'Action')

    def __init__(self, Event_To_Broadcast='UNINITIALIZED STRING', Event_Type=Action_Event_Type_Enum.INDIVIDUAL, Threshold=0, **kwargs):
        super(Action, self).__init__(**kwargs)
        self.Event_To_Broadcast = Event_To_Broadcast
        self.Event_Type = (Event_Type.name if isinstance(Event_Type, Enum) else Event_Type)
        self.Threshold = Threshold



class AgeAndProbability(BaseCampaign):
    _definition = {
        'Age': {
            'default': 0,
            'description': "An array of ages (in days). In IVCalendar, there is a list of actual interventions where the distribution is dependent on whether the individual's age matches the next date in the calendar.",
            'max': 45625,
            'min': 0,
            'type': 'float',
        },
        'Probability': {
            'default': 0,
            'description': 'The probability of an individual receiving the list of actual interventions at the corresponding age.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'AgeAndProbability',
    }
    _validator = ClassValidator(_definition, 'AgeAndProbability')

    def __init__(self, Age=0, Probability=0, **kwargs):
        super(AgeAndProbability, self).__init__(**kwargs)
        self.Age = Age
        self.Probability = Probability



class AgeRange(BaseCampaign):
    _definition = {
        'Max': {
            'default': 125,
            'description': 'This parameter determines the maximum age, in years for individuals to be included in the targeted population. An individual is considered in range if their age is greater than or equal to the minimum age and less than the maximum age, in floating point years of age.',
            'max': 125,
            'min': 0,
            'type': 'float',
        },
        'Min': {
            'default': 0,
            'description': 'This parameter determines the minimum age, in years for individuals to be included in the targeted population. An individual is considered in range if their age is greater than or equal to the minimum age and less than the maximum age, in floating point years of age.',
            'max': 125,
            'min': 0,
            'type': 'float',
        },
        'class': 'AgeRange',
    }
    _validator = ClassValidator(_definition, 'AgeRange')

    def __init__(self, Max=125, Min=0, **kwargs):
        super(AgeRange, self).__init__(**kwargs)
        self.Max = Max
        self.Min = Min



class Choice(BaseCampaign):
    _definition = {
        'Ages': {
            'ascending': 1,
            'default': [],
            'description': 'TBD',
            'max': 125,
            'min': 0,
            'type': 'Vector Float',
        },
        'Broadcast_Event': {
            'default': 'UNINITIALIZED STRING',
            'description': 'TBD',
            'type': 'string',
        },
        'Parity': {
            'ascending': 1,
            'default': [],
            'description': 'TBD',
            'max': 20,
            'min': 0,
            'type': 'Vector Uint32',
        },
        'ProbabilityMatrix': {
            'default': [],
            'description': 'TBD',
            'max': 1,
            'min': 0,
            'type': 'Vector3d Float',
        },
        'Years': {
            'ascending': 1,
            'default': [],
            'description': 'TBD',
            'max': 2200,
            'min': 1800,
            'type': 'Vector Float',
        },
        'class': 'Choice',
    }
    _validator = ClassValidator(_definition, 'Choice')

    def __init__(self, Ages=[], Broadcast_Event='UNINITIALIZED STRING', Parity=[], ProbabilityMatrix=[], Years=[], **kwargs):
        super(Choice, self).__init__(**kwargs)
        self.Ages = Ages
        self.Broadcast_Event = Broadcast_Event
        self.Parity = Parity
        self.ProbabilityMatrix = ProbabilityMatrix
        self.Years = Years



class Filter(BaseCampaign):
    _definition = {
        'Multipliers': {
            'ascending': 0,
            'default': [],
            'description': 'TBD',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'Vector Float',
        },
        'Properties': {
            'default': [],
            'description': 'TBD',
            'type': 'Vector Constrained String',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'class': 'Filter',
    }
    _validator = ClassValidator(_definition, 'Filter')

    def __init__(self, Multipliers=[], Properties=[], **kwargs):
        super(Filter, self).__init__(**kwargs)
        self.Multipliers = Multipliers
        self.Properties = Properties



class IncidenceCounter(BaseCampaign):
    _definition = {
        'Count_Events_For_Num_Timesteps': {
            'default': 1,
            'description': 'If set to true (1), then the waning effect values, as specified in the Effect_List list of objects for the WaningEffectCombo classes, are added together. If set to false (0), the waning effect values are multiplied. The resulting waning effect value cannot be greater than 1.',
            'max': 2147480000.0,
            'min': 1,
            'type': 'integer',
        },
        'Demographic_Coverage': {
            'default': 1,
            'description': 'The fraction of individuals in the target demographic that will receive this intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Trigger_Condition_List': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': [],
            'description': 'A list of events that will trigger an intervention.',
            'type': 'Vector Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'class': 'IncidenceCounter',
    }
    _validator = ClassValidator(_definition, 'IncidenceCounter')

    def __init__(self, Count_Events_For_Num_Timesteps=1, Demographic_Coverage=1, Node_Property_Restrictions=[], Property_Restrictions=[], Property_Restrictions_Within_Node=[], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=IncidenceCounter_Target_Demographic_Enum.Everyone, Target_Gender=IncidenceCounter_Target_Gender_Enum.All, Target_Residents_Only=False, Trigger_Condition_List=[], **kwargs):
        super(IncidenceCounter, self).__init__(**kwargs)
        self.Count_Events_For_Num_Timesteps = Count_Events_For_Num_Timesteps
        self.Demographic_Coverage = Demographic_Coverage
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Trigger_Condition_List = Trigger_Condition_List



class IncidenceCounterSurveillance(BaseCampaign):
    _definition = {
        'Count_Events_For_Num_Timesteps': {
            'default': 1,
            'description': 'If set to true (1), then the waning effect values, as specified in the Effect_List list of objects for the WaningEffectCombo classes, are added together. If set to false (0), the waning effect values are multiplied. The resulting waning effect value cannot be greater than 1.',
            'max': 2147480000.0,
            'min': 1,
            'type': 'integer',
        },
        'Counter_Event_Type': {
            'default': 'INDIVIDUAL',
            'description': 'Type of events that can be included in **Trigger_Condition_List**. Possible values are: COORDINATOR, INDIVIDUAL, NODE.',
            'enum': ['INDIVIDUAL', 'NODE', 'COORDINATOR'],
            'type': 'enum',
        },
        'Counter_Period': {
            'default': 1,
            'description': 'When **Counter_Type** is set to PERIODIC, this is the counter period (in days).',
            'max': 1000,
            'min': 1,
            'type': 'float',
        },
        'Counter_Type': {
            'default': 'PERIODIC',
            'description': 'Counter type. Possible values are: PERIODIC.',
            'enum': ['PERIODIC'],
            'type': 'enum',
        },
        'Demographic_Coverage': {
            'default': 1,
            'description': 'The fraction of individuals in the target demographic that will receive this intervention.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Node_Property_Restrictions': {
            'description': 'A list of the NodeProperty key:value pairs, as defined in the demographics file, that nodes must have to be targeted by the intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::NodeProperties.*.Property',
                    'description': 'Node Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::NodeProperties.*.Values',
                    'description': 'Node Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Property_Restrictions': {
            'default': [],
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'Dynamic String Set',
            'value_source': "'<demographics>::*.Individual_Properties.*.Property':'<demographics>::*.Individual_Properties.*.Values'",
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Sim_Types': ['*'],
        'Target_Age_Max': {
            'default': 9.3228e+35,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The upper end of ages targeted for an intervention, in years.',
            'max': 9.3228e+35,
            'min': 0,
            'type': 'float',
        },
        'Target_Age_Min': {
            'default': 0,
            'depends-on': {
                'Target_Demographic': 'ExplicitAgeRanges,ExplicitAgeRangesAndGender',
            },
            'description': 'The lower end of ages targeted for an intervention, in years.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Target_Demographic': {
            'default': 'Everyone',
            'description': 'The target demographic group.',
            'enum': ['Everyone', 'ExplicitAgeRanges', 'ExplicitAgeRangesAndGender', 'ExplicitGender', 'PossibleMothers', 'ArrivingAirTravellers', 'DepartingAirTravellers', 'ArrivingRoadTravellers', 'DepartingRoadTravellers', 'AllArrivingTravellers', 'AllDepartingTravellers', 'ExplicitDiseaseState', 'Pregnant'],
            'type': 'enum',
        },
        'Target_Gender': {
            'default': 'All',
            'description': 'Specifies the gender restriction for the intervention.',
            'enum': ['All', 'Male', 'Female'],
            'type': 'enum',
        },
        'Target_Residents_Only': {
            'default': 0,
            'description': 'When set to true (1), the intervention is only distributed to individuals that began the simulation in the node (i.e. those that claim the node as their residence).',
            'type': 'bool',
        },
        'Trigger_Condition_List': {
            'default': [],
            'description': 'The list of events to count. The list cannot be empty.',
            'type': 'Vector String',
        },
        'class': 'IncidenceCounterSurveillance',
    }
    _validator = ClassValidator(_definition, 'IncidenceCounterSurveillance')

    def __init__(self, Count_Events_For_Num_Timesteps=1, Counter_Event_Type=IncidenceCounterSurveillance_Counter_Event_Type_Enum.INDIVIDUAL, Counter_Period=1, Counter_Type=IncidenceCounterSurveillance_Counter_Type_Enum.PERIODIC, Demographic_Coverage=1, Node_Property_Restrictions=[], Property_Restrictions=[], Property_Restrictions_Within_Node=[], Sim_Types=['*'], Target_Age_Max=9.3228e+35, Target_Age_Min=0, Target_Demographic=IncidenceCounterSurveillance_Target_Demographic_Enum.Everyone, Target_Gender=IncidenceCounterSurveillance_Target_Gender_Enum.All, Target_Residents_Only=False, Trigger_Condition_List=[], **kwargs):
        super(IncidenceCounterSurveillance, self).__init__(**kwargs)
        self.Count_Events_For_Num_Timesteps = Count_Events_For_Num_Timesteps
        self.Counter_Event_Type = (Counter_Event_Type.name if isinstance(Counter_Event_Type, Enum) else Counter_Event_Type)
        self.Counter_Period = Counter_Period
        self.Counter_Type = (Counter_Type.name if isinstance(Counter_Type, Enum) else Counter_Type)
        self.Demographic_Coverage = Demographic_Coverage
        self.Node_Property_Restrictions = Node_Property_Restrictions
        self.Property_Restrictions = Property_Restrictions
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Sim_Types = Sim_Types
        self.Target_Age_Max = Target_Age_Max
        self.Target_Age_Min = Target_Age_Min
        self.Target_Demographic = (Target_Demographic.name if isinstance(Target_Demographic, Enum) else Target_Demographic)
        self.Target_Gender = (Target_Gender.name if isinstance(Target_Gender, Enum) else Target_Gender)
        self.Target_Residents_Only = Target_Residents_Only
        self.Trigger_Condition_List = Trigger_Condition_List



class InterpolatedValueMap(BaseCampaign):
    _definition = {
        'Times': {
            'ascending': 1,
            'default': [],
            'description': 'An array of years.',
            'max': 999999,
            'min': 0,
            'type': 'Vector Float',
        },
        'Values': {
            'ascending': 0,
            'default': [],
            'description': 'An array of values to match the defined Times.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'Vector Float',
        },
        'class': 'InterpolatedValueMap',
    }
    _validator = ClassValidator(_definition, 'InterpolatedValueMap')

    def __init__(self, Times=[], Values=[], **kwargs):
        super(InterpolatedValueMap, self).__init__(**kwargs)
        self.Times = Times
        self.Values = Values



class LarvalHabitatMultiplier(BaseCampaign):
    _definition = {
        'LarvalHabitatMultiplier': {
            'default': [],
            'description': 'The value by which to scale the larval habitat availability specified in the configuration file with Larval_Habitat_Types across all habitat types, for specific habitat types, or for specific mosquito species within each habitat type.',
            'item_type': 'LarvalHabitatMultiplierSpec',
            'type': 'Vector',
        },
        'class': 'LarvalHabitatMultiplier',
    }
    _validator = ClassValidator(_definition, 'LarvalHabitatMultiplier')

    def __init__(self, LarvalHabitatMultiplier=[], **kwargs):
        super(LarvalHabitatMultiplier, self).__init__(**kwargs)
        self.LarvalHabitatMultiplier = LarvalHabitatMultiplier



class LarvalHabitatMultiplierSpec(BaseCampaign):
    _definition = {
        'Factor': {
            'default': 1,
            'description': 'The value by which to scale the larval habitat availability',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Habitat': {
            'default': 'TEMPORARY_RAINFALL',
            'description': 'The name of the habitat for which to specify a larval habitat multiplier',
            'enum': ['TEMPORARY_RAINFALL', 'WATER_VEGETATION', 'HUMAN_POPULATION', 'CONSTANT', 'BRACKISH_SWAMP', 'MARSHY_STREAM', 'LINEAR_SPLINE', 'ALL_HABITATS'],
            'type': 'enum',
        },
        'Species': {
            'default': 'UNINITIALIZED STRING',
            'description': 'The name of the species for which to specify a larval habitat multiplier',
            'type': 'string',
        },
        'class': 'LarvalHabitatMultiplierSpec',
    }
    _validator = ClassValidator(_definition, 'LarvalHabitatMultiplierSpec')

    def __init__(self, Factor=1, Habitat=LarvalHabitatMultiplierSpec_Habitat_Enum.TEMPORARY_RAINFALL, Species='UNINITIALIZED STRING', **kwargs):
        super(LarvalHabitatMultiplierSpec, self).__init__(**kwargs)
        self.Factor = Factor
        self.Habitat = (Habitat.name if isinstance(Habitat, Enum) else Habitat)
        self.Species = Species



class NodeIdAndCoverage(BaseCampaign):
    _definition = {
        'Coverage': {
            'default': 0,
            'description': 'TBD',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Node_Id': {
            'default': 0,
            'description': 'TBD',
            'max': 999999,
            'min': 0,
            'type': 'integer',
        },
        'class': 'NodeIdAndCoverage',
    }
    _validator = ClassValidator(_definition, 'NodeIdAndCoverage')

    def __init__(self, Coverage=0, Node_Id=0, **kwargs):
        super(NodeIdAndCoverage, self).__init__(**kwargs)
        self.Coverage = Coverage
        self.Node_Id = Node_Id



class RangeThreshold(BaseCampaign):
    _definition = {
        'Event': {
            'Built-in': ['Births', 'EveryUpdate', 'EveryTimeStep', 'NewInfectionEvent', 'TBActivation', 'NewClinicalCase', 'NewSevereCase', 'DiseaseDeaths', 'OpportunisticInfectionDeath', 'NonDiseaseDeaths', 'TBActivationSmearPos', 'TBActivationSmearNeg', 'TBActivationExtrapulm', 'TBActivationPostRelapse', 'TBPendingRelapse', 'TBActivationPresymptomatic', 'TestPositiveOnSmear', 'ProviderOrdersTBTest', 'TBTestPositive', 'TBTestNegative', 'TBTestDefault', 'TBRestartHSB', 'TBMDRTestPositive', 'TBMDRTestNegative', 'TBMDRTestDefault', 'TBFailedDrugRegimen', 'TBRelapseAfterDrugRegimen', 'TBStartDrugRegimen', 'TBStopDrugRegimen', 'PropertyChange', 'STIDebut', 'StartedART', 'StoppedART', 'InterventionDisqualified', 'HIVNewlyDiagnosed', 'GaveBirth', 'Pregnant', 'Emigrating', 'Immigrating', 'HIVTestedNegative', 'HIVTestedPositive', 'NewlySymptomatic', 'SymptomaticCleared', 'TwelveWeeksPregnant', 'FourteenWeeksPregnant', 'SixWeeksOld', 'EighteenMonthsOld', 'STIPreEmigrating', 'STIPostImmigrating', 'STINewInfection', 'NewExternalHIVInfection', 'NodePropertyChange', 'HappyBirthday', 'EnteredRelationship', 'ExitedRelationship', 'FirstCoitalAct', 'ExposureComplete', 'StartBeingPossibleMother', 'EndBeingPossibleMother'],
            'default': '',
            'description': 'The user-defined name of the diagnostic event.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Individual_Events.*' or Built-in",
        },
        'High': {
            'default': 0,
            'description': 'The high end of the diagnostic level.',
            'max': 2000,
            'min': 0,
            'type': 'float',
        },
        'Low': {
            'default': 0,
            'description': 'The low end of the diagnostic level.',
            'max': 2000,
            'min': 0,
            'type': 'float',
        },
        'class': 'RangeThreshold',
    }
    _validator = ClassValidator(_definition, 'RangeThreshold')

    def __init__(self, Event='', High=0, Low=0, **kwargs):
        super(RangeThreshold, self).__init__(**kwargs)
        self.Event = Event
        self.High = High
        self.Low = Low



class Responder(BaseCampaign):
    _definition = {
        'Action_List': {
            'default': [],
            'description': 'List (array) of JSON objects, including the values specified in the following parameters: * Threshold, * Event_To_Broadcast.',
            'item_type': 'Action',
            'type': 'Vector',
        },
        'Threshold_Type': {
            'default': 'COUNT',
            'description': 'Threshold type. Possible values are COUNT and PERCENTAGE.',
            'enum': ['COUNT', 'PERCENTAGE', 'PERCENTAGE_EVENTS'],
            'type': 'enum',
        },
        'class': 'Responder',
    }
    _validator = ClassValidator(_definition, 'Responder')

    def __init__(self, Action_List=[], Threshold_Type=Responder_Threshold_Type_Enum.COUNT, **kwargs):
        super(Responder, self).__init__(**kwargs)
        self.Action_List = Action_List
        self.Threshold_Type = (Threshold_Type.name if isinstance(Threshold_Type, Enum) else Threshold_Type)



class ResponderSurveillance(BaseCampaign):
    _definition = {
        'Action_List': {
            'default': [],
            'description': 'List (array) of JSON objects, including the values specified in the following parameters: * Threshold, * Event_To_Broadcast.',
            'item_type': 'Action',
            'type': 'Vector',
        },
        'Percentage_Events_To_Count': {
            'default': [],
            'depends-on': {
                'Threshold_Type': 'PERCENTAGE_EVENTS',
            },
            'description': 'TBD',
            'type': 'Vector String',
        },
        'Responded_Event': {
            'default': '',
            'description': 'A coordinator event, defined in **Custom_Coordinator_Events**, that is broadcast if **Responder** responded.',
            'type': 'Constrained String',
            'value_source': "'<configuration>:Custom_Coordinator_Events.*' or Built-in",
        },
        'Threshold_Type': {
            'default': 'COUNT',
            'description': 'Threshold type. Possible values are COUNT and PERCENTAGE.',
            'enum': ['COUNT', 'PERCENTAGE', 'PERCENTAGE_EVENTS'],
            'type': 'enum',
        },
        'class': 'ResponderSurveillance',
    }
    _validator = ClassValidator(_definition, 'ResponderSurveillance')

    def __init__(self, Action_List=[], Percentage_Events_To_Count=[], Responded_Event='', Threshold_Type=ResponderSurveillance_Threshold_Type_Enum.COUNT, **kwargs):
        super(ResponderSurveillance, self).__init__(**kwargs)
        self.Action_List = Action_List
        self.Percentage_Events_To_Count = Percentage_Events_To_Count
        self.Responded_Event = Responded_Event
        self.Threshold_Type = (Threshold_Type.name if isinstance(Threshold_Type, Enum) else Threshold_Type)



class TargetedDistribution(BaseCampaign):
    _definition = {
        'Age_Ranges_Years': {
            'default': [],
            'description': 'A list of age ranges that individuals must be in to qualify for an intervention. Each age range is a JSON object with a minimum and a maximum property.',
            'item_type': 'AgeRange',
            'type': 'Vector',
        },
        'End_Day': {
            'default': 3.40282e+38,
            'description': 'The day to stop distributing the intervention. No interventions are distributed on this day or going forward.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'Num_Targeted': {
            'ascending': 0,
            'default': [],
            'description': 'The number of individuals to target with the intervention.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'Vector Int',
        },
        'Num_Targeted_Females': {
            'ascending': 0,
            'default': [],
            'description': 'The number of female individuals to distribute interventions to during this time period.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'Vector Int',
        },
        'Num_Targeted_Males': {
            'ascending': 0,
            'default': [],
            'description': 'The number of male individuals to distribute interventions to during this time period.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'Vector Int',
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Start_Day': {
            'default': 0,
            'description': 'The day to start distributing the intervention.',
            'max': 3.40282e+38,
            'min': 0,
            'type': 'float',
        },
        'class': 'TargetedDistribution',
    }
    _validator = ClassValidator(_definition, 'TargetedDistribution')

    def __init__(self, Age_Ranges_Years=[], End_Day=3.40282e+38, Num_Targeted=[], Num_Targeted_Females=[], Num_Targeted_Males=[], Property_Restrictions_Within_Node=[], Start_Day=0, **kwargs):
        super(TargetedDistribution, self).__init__(**kwargs)
        self.Age_Ranges_Years = Age_Ranges_Years
        self.End_Day = End_Day
        self.Num_Targeted = Num_Targeted
        self.Num_Targeted_Females = Num_Targeted_Females
        self.Num_Targeted_Males = Num_Targeted_Males
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Start_Day = Start_Day



class TargetedDistributionHIV(BaseCampaign):
    _definition = {
        'Age_Ranges_Years': {
            'default': [],
            'description': 'A list of age ranges that individuals must be in to qualify for an intervention. Each age range is a JSON object with a minimum and a maximum property.',
            'item_type': 'AgeRange',
            'type': 'Vector',
        },
        'End_Year': {
            'default': 2200,
            'description': 'The year to stop distributing the intervention.',
            'max': 2200,
            'min': 1900,
            'type': 'float',
        },
        'Num_Targeted': {
            'ascending': 0,
            'default': [],
            'description': 'The number of individuals to target with the intervention.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'Vector Int',
        },
        'Num_Targeted_Females': {
            'ascending': 0,
            'default': [],
            'description': 'The number of female individuals to distribute interventions to during this time period.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'Vector Int',
        },
        'Num_Targeted_Males': {
            'ascending': 0,
            'default': [],
            'description': 'The number of male individuals to distribute interventions to during this time period.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'Vector Int',
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Start_Year': {
            'default': 1900,
            'description': 'The year to start distributing the intervention.',
            'max': 2200,
            'min': 1900,
            'type': 'float',
        },
        'Target_Disease_State': {
            'default': [],
            'description': 'A two-dimensional array of particular disease states. To qualify for the intervention, an individual must have only one of the targeted disease states. An individual must have all of the disease states in the inner array.',
            'type': 'Vector2d String',
        },
        'Target_Disease_State_Has_Intervention_Name': {
            'default': '',
            'description': 'The name of the intervention to look for in an individual when using Has_Intervention or Not_have_Intervention in Target_Disease_State.',
            'type': 'string',
        },
        'class': 'TargetedDistributionHIV',
    }
    _validator = ClassValidator(_definition, 'TargetedDistributionHIV')

    def __init__(self, Age_Ranges_Years=[], End_Year=2200, Num_Targeted=[], Num_Targeted_Females=[], Num_Targeted_Males=[], Property_Restrictions_Within_Node=[], Start_Year=1900, Target_Disease_State=[], Target_Disease_State_Has_Intervention_Name='', **kwargs):
        super(TargetedDistributionHIV, self).__init__(**kwargs)
        self.Age_Ranges_Years = Age_Ranges_Years
        self.End_Year = End_Year
        self.Num_Targeted = Num_Targeted
        self.Num_Targeted_Females = Num_Targeted_Females
        self.Num_Targeted_Males = Num_Targeted_Males
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Start_Year = Start_Year
        self.Target_Disease_State = Target_Disease_State
        self.Target_Disease_State_Has_Intervention_Name = Target_Disease_State_Has_Intervention_Name



class TargetedDistributionSTI(BaseCampaign):
    _definition = {
        'Age_Ranges_Years': {
            'default': [],
            'description': 'A list of age ranges that individuals must be in to qualify for an intervention. Each age range is a JSON object with a minimum and a maximum property.',
            'item_type': 'AgeRange',
            'type': 'Vector',
        },
        'End_Year': {
            'default': 2200,
            'description': 'The year to stop distributing the intervention.',
            'max': 2200,
            'min': 1900,
            'type': 'float',
        },
        'Num_Targeted': {
            'ascending': 0,
            'default': [],
            'description': 'The number of individuals to target with the intervention.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'Vector Int',
        },
        'Num_Targeted_Females': {
            'ascending': 0,
            'default': [],
            'description': 'The number of female individuals to distribute interventions to during this time period.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'Vector Int',
        },
        'Num_Targeted_Males': {
            'ascending': 0,
            'default': [],
            'description': 'The number of male individuals to distribute interventions to during this time period.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'Vector Int',
        },
        'Property_Restrictions_Within_Node': {
            'description': 'A list of the IndividualProperty key:value pairs, as defined in the demographics file, that individuals must have to be targeted by this intervention.',
            'type': 'list',
            'item_type': {
                'type': 'dict',
                'key_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Property',
                    'description': 'Individual Property Key from demographics file.',
                    'type': 'Constrained String',
                },
                'value_type': {
                    'constraints': '<demographics>::*.Individual_Properties.*.Values',
                    'description': 'Individual Property Value from demographics file.',
                    'type': 'String',
                },
            },
            'default': [],
        },
        'Start_Year': {
            'default': 1900,
            'description': 'The year to start distributing the intervention.',
            'max': 2200,
            'min': 1900,
            'type': 'float',
        },
        'class': 'TargetedDistributionSTI',
    }
    _validator = ClassValidator(_definition, 'TargetedDistributionSTI')

    def __init__(self, Age_Ranges_Years=[], End_Year=2200, Num_Targeted=[], Num_Targeted_Females=[], Num_Targeted_Males=[], Property_Restrictions_Within_Node=[], Start_Year=1900, **kwargs):
        super(TargetedDistributionSTI, self).__init__(**kwargs)
        self.Age_Ranges_Years = Age_Ranges_Years
        self.End_Year = End_Year
        self.Num_Targeted = Num_Targeted
        self.Num_Targeted_Females = Num_Targeted_Females
        self.Num_Targeted_Males = Num_Targeted_Males
        self.Property_Restrictions_Within_Node = Property_Restrictions_Within_Node
        self.Start_Year = Start_Year



class WaningEffectBox(BaseCampaign):
    _definition = {
        'Box_Duration': {
            'default': 100,
            'description': 'Box duration of effect in days.',
            'max': 100000,
            'min': 0,
            'type': 'float',
        },
        'Initial_Effect': {
            'default': 1,
            'description': 'Initial strength of the effect. The effect decays over time.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'WaningEffectBox',
    }
    _validator = ClassValidator(_definition, 'WaningEffectBox')

    def __init__(self, Box_Duration=100, Initial_Effect=1, **kwargs):
        super(WaningEffectBox, self).__init__(**kwargs)
        self.Box_Duration = Box_Duration
        self.Initial_Effect = Initial_Effect



class WaningEffectBoxExponential(BaseCampaign):
    _definition = {
        'Box_Duration': {
            'default': 100,
            'description': 'Box duration of effect in days.',
            'max': 100000,
            'min': 0,
            'type': 'float',
        },
        'Decay_Time_Constant': {
            'default': 100,
            'description': 'The exponential decay length, in days.',
            'max': 100000,
            'min': 0,
            'type': 'float',
        },
        'Initial_Effect': {
            'default': 1,
            'description': 'Initial strength of the effect. The effect decays over time.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'WaningEffectBoxExponential',
    }
    _validator = ClassValidator(_definition, 'WaningEffectBoxExponential')

    def __init__(self, Box_Duration=100, Decay_Time_Constant=100, Initial_Effect=1, **kwargs):
        super(WaningEffectBoxExponential, self).__init__(**kwargs)
        self.Box_Duration = Box_Duration
        self.Decay_Time_Constant = Decay_Time_Constant
        self.Initial_Effect = Initial_Effect



class WaningEffectCombo(BaseCampaign):
    _definition = {
        'Add_Effects': {
            'default': 0,
            'description': 'The Add_Effects parameter allows you to combine multiple effects from the waning effect classes. If set to true (1), then the waning effect values from the different waning effect objects are added together. If set to false (0), the waning effect values are multiplied. The resulting waning effect value must be greater than 1.',
            'type': 'bool',
        },
        'Effect_List': {
            'description': 'A list of nested JSON objects to indicate how the intervention efficacy wanes over time.',
            '<WaningEffect Value>': {
                'type_name': 'idmType:WaningEffect',
                'type_schema': {
                    'base': 'interventions.idmType.WaningEffect',
                },
            },
        },
        'Expires_When_All_Expire': {
            'default': 0,
            'description': 'If set to true (1), then all of the effects, as specified in the Effect_List parameter, must expire for the efficacy of the intervention to expire. If set to false (0), then the efficacy of the intervention will expire as soon as one of the parameters expires.',
            'type': 'bool',
        },
        'class': 'WaningEffectCombo',
    }
    _validator = ClassValidator(_definition, 'WaningEffectCombo')

    def __init__(self, Add_Effects=False, Effect_List=None, Expires_When_All_Expire=False, **kwargs):
        super(WaningEffectCombo, self).__init__(**kwargs)
        self.Add_Effects = Add_Effects
        self.Effect_List = Effect_List
        self.Expires_When_All_Expire = Expires_When_All_Expire



class WaningEffectConstant(BaseCampaign):
    _definition = {
        'Initial_Effect': {
            'default': 1,
            'description': 'Initial strength of the effect. The effect decays over time.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'WaningEffectConstant',
    }
    _validator = ClassValidator(_definition, 'WaningEffectConstant')

    def __init__(self, Initial_Effect=1, **kwargs):
        super(WaningEffectConstant, self).__init__(**kwargs)
        self.Initial_Effect = Initial_Effect



class WaningEffectExponential(BaseCampaign):
    _definition = {
        'Decay_Time_Constant': {
            'default': 100,
            'description': 'The exponential decay length, in days.',
            'max': 100000,
            'min': 0,
            'type': 'float',
        },
        'Initial_Effect': {
            'default': 1,
            'description': 'Initial strength of the effect. The effect decays over time.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'WaningEffectExponential',
    }
    _validator = ClassValidator(_definition, 'WaningEffectExponential')

    def __init__(self, Decay_Time_Constant=100, Initial_Effect=1, **kwargs):
        super(WaningEffectExponential, self).__init__(**kwargs)
        self.Decay_Time_Constant = Decay_Time_Constant
        self.Initial_Effect = Initial_Effect



class WaningEffectMapCount(BaseCampaign):
    _definition = {
        'Durability_Map': {
            'description': 'The time, in days, since the intervention was distributed and a multiplier for the Initial_Effect.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Initial_Effect': {
            'default': 1,
            'description': 'Initial strength of the effect. The effect decays over time.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'WaningEffectMapCount',
    }
    _validator = ClassValidator(_definition, 'WaningEffectMapCount')

    def __init__(self, Durability_Map=None, Initial_Effect=1, **kwargs):
        super(WaningEffectMapCount, self).__init__(**kwargs)
        self.Durability_Map = Durability_Map
        self.Initial_Effect = Initial_Effect



class WaningEffectMapLinear(BaseCampaign):
    _definition = {
        'Durability_Map': {
            'description': 'The time, in days, since the intervention was distributed and a multiplier for the Initial_Effect.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Expire_At_Durability_Map_End': {
            'default': 0,
            'description': 'Set to 1 to let the intervention expire when the end of the map is reached.',
            'type': 'bool',
        },
        'Initial_Effect': {
            'default': 1,
            'description': 'Initial strength of the effect. The effect decays over time.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Reference_Timer': {
            'default': 0,
            'description': 'Timestamp at which linear-map should be anchored.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'integer',
        },
        'class': 'WaningEffectMapLinear',
    }
    _validator = ClassValidator(_definition, 'WaningEffectMapLinear')

    def __init__(self, Durability_Map=None, Expire_At_Durability_Map_End=False, Initial_Effect=1, Reference_Timer=0, **kwargs):
        super(WaningEffectMapLinear, self).__init__(**kwargs)
        self.Durability_Map = Durability_Map
        self.Expire_At_Durability_Map_End = Expire_At_Durability_Map_End
        self.Initial_Effect = Initial_Effect
        self.Reference_Timer = Reference_Timer



class WaningEffectMapLinearAge(BaseCampaign):
    _definition = {
        'Durability_Map': {
            'description': 'The time, in days, since the intervention was distributed and a multiplier for the Initial_Effect.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Initial_Effect': {
            'default': 1,
            'description': 'Initial strength of the effect. The effect decays over time.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'WaningEffectMapLinearAge',
    }
    _validator = ClassValidator(_definition, 'WaningEffectMapLinearAge')

    def __init__(self, Durability_Map=None, Initial_Effect=1, **kwargs):
        super(WaningEffectMapLinearAge, self).__init__(**kwargs)
        self.Durability_Map = Durability_Map
        self.Initial_Effect = Initial_Effect



class WaningEffectMapLinearSeasonal(BaseCampaign):
    _definition = {
        'Durability_Map': {
            'description': 'The time, in days, since the intervention was distributed and a multiplier for the Initial_Effect.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Initial_Effect': {
            'default': 1,
            'description': 'Initial strength of the effect. The effect decays over time.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'WaningEffectMapLinearSeasonal',
    }
    _validator = ClassValidator(_definition, 'WaningEffectMapLinearSeasonal')

    def __init__(self, Durability_Map=None, Initial_Effect=1, **kwargs):
        super(WaningEffectMapLinearSeasonal, self).__init__(**kwargs)
        self.Durability_Map = Durability_Map
        self.Initial_Effect = Initial_Effect



class WaningEffectMapPiecewise(BaseCampaign):
    _definition = {
        'Durability_Map': {
            'description': 'The time, in days, since the intervention was distributed and a multiplier for the Initial_Effect.',
            'type': 'object',
            'subclasses': 'InterpolatedValueMap',
        },
        'Expire_At_Durability_Map_End': {
            'default': 0,
            'description': 'Set to 1 to let the intervention expire when the end of the map is reached.',
            'type': 'bool',
        },
        'Initial_Effect': {
            'default': 1,
            'description': 'Initial strength of the effect. The effect decays over time.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'Reference_Timer': {
            'default': 0,
            'description': 'Timestamp at which linear-map should be anchored.',
            'max': 2147480000.0,
            'min': 0,
            'type': 'integer',
        },
        'class': 'WaningEffectMapPiecewise',
    }
    _validator = ClassValidator(_definition, 'WaningEffectMapPiecewise')

    def __init__(self, Durability_Map=None, Expire_At_Durability_Map_End=False, Initial_Effect=1, Reference_Timer=0, **kwargs):
        super(WaningEffectMapPiecewise, self).__init__(**kwargs)
        self.Durability_Map = Durability_Map
        self.Expire_At_Durability_Map_End = Expire_At_Durability_Map_End
        self.Initial_Effect = Initial_Effect
        self.Reference_Timer = Reference_Timer



class WaningEffectRandomBox(BaseCampaign):
    _definition = {
        'Expected_Discard_Time': {
            'default': 100,
            'description': 'The mean time, in days, of an exponential distribution of the duration of the effect of an intervention (such as a vaccine or bed net).',
            'max': 100000,
            'min': 0,
            'type': 'float',
        },
        'Initial_Effect': {
            'default': 1,
            'description': 'Initial strength of the effect. The effect decays over time.',
            'max': 1,
            'min': 0,
            'type': 'float',
        },
        'class': 'WaningEffectRandomBox',
    }
    _validator = ClassValidator(_definition, 'WaningEffectRandomBox')

    def __init__(self, Expected_Discard_Time=100, Initial_Effect=1, **kwargs):
        super(WaningEffectRandomBox, self).__init__(**kwargs)
        self.Expected_Discard_Time = Expected_Discard_Time
        self.Initial_Effect = Initial_Effect

###################################
# Our classes
###################################


class Campaign(BaseCampaign):
    _definition = {"Campaign_Name": "Empty Campaign", "Events": [], "Use_Defaults": {"default": True, "type": "bool"}}
    _validator = ClassValidator(_definition, 'Campaign')

    def __init__(self, Campaign_Name="Empty Campaign", Use_Defaults=True, Events=None, **kwargs):
        super(Campaign, self).__init__(**kwargs)
        self.Campaign_Name = Campaign_Name
        self.Use_Defaults = Use_Defaults
        self.Events = Events or []

    def add_campaign_event(self, ce):
        self.Events.append(ce)
