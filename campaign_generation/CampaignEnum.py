from enum import Enum, auto


class CalendarEventCoordinator_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class CalendarEventCoordinator_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class CommunityHealthWorkerEventCoordinator_Initial_Amount_Distribution_Enum(
    Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class CommunityHealthWorkerEventCoordinator_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class CommunityHealthWorkerEventCoordinator_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class CoverageByNodeEventCoordinator_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class CoverageByNodeEventCoordinator_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class DelayEventCoordinator_Delay_Period_Distribution_Enum(Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class GroupInterventionDistributionEventCoordinator_Target_Demographic_Enum(
    Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class GroupInterventionDistributionEventCoordinator_Target_Disease_State_Enum(
    Enum):
    Everyone = auto()
    Infected = auto()
    ActiveInfection = auto()
    LatentInfection = auto()
    MDR = auto()
    TreatmentNaive = auto()
    HasFailedTreatment = auto()
    HIVNegative = auto()
    ActiveHadTreatment = auto()


class GroupInterventionDistributionEventCoordinator_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class GroupInterventionDistributionEventCoordinatorHIV_Target_Demographic_Enum(
    Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class GroupInterventionDistributionEventCoordinatorHIV_Target_Gender_Enum(Enum
    ):
    All = auto()
    Male = auto()
    Female = auto()


class ReferenceTrackingEventCoordinator_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class ReferenceTrackingEventCoordinator_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class ReferenceTrackingEventCoordinatorHIV_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class ReferenceTrackingEventCoordinatorHIV_Target_Disease_State_Enum(Enum):
    Everyone = auto()
    HIV_Positive = auto()
    HIV_Negative = auto()
    Tested_Positive = auto()
    Tested_Negative = auto()
    Not_Tested_Or_Tested_Negative = auto()


class ReferenceTrackingEventCoordinatorHIV_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class StandardInterventionDistributionEventCoordinator_Target_Demographic_Enum(
    Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class StandardInterventionDistributionEventCoordinator_Target_Gender_Enum(Enum
    ):
    All = auto()
    Male = auto()
    Female = auto()


class TriggeredEventCoordinator_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class TriggeredEventCoordinator_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class ActiveDiagnostic_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class AdherentDrug_Dosing_Type_Enum(Enum):
    SingleDose = auto()
    FullTreatmentCourse = auto()
    Prophylaxis = auto()
    SingleDoseWhenSymptom = auto()
    FullTreatmentWhenSymptom = auto()
    SingleDoseParasiteDetect = auto()
    FullTreatmentParasiteDetect = auto()
    SingleDoseNewDetectionTech = auto()
    FullTreatmentNewDetectionTech = auto()


class AdherentDrug_Non_Adherence_Options_Enum(Enum):
    NEXT_UPDATE = auto()
    NEXT_DOSAGE_TIME = auto()
    LOST_TAKE_NEXT = auto()
    STOP = auto()


class AntiTBDrug_Drug_Type_Enum(Enum):
    DOTS = auto()
    DOTSImproved = auto()
    EmpiricTreatment = auto()
    FirstLineCombo = auto()
    SecondLineCombo = auto()
    ThirdLineCombo = auto()
    LatentTreatment = auto()


class AntiTBDrug_Durability_Profile_Enum(Enum):
    FIXED_DURATION_CONSTANT_EFFECT = auto()
    CONCENTRATION_VERSUS_TIME = auto()


class AntimalarialDrug_Dosing_Type_Enum(Enum):
    SingleDose = auto()
    FullTreatmentCourse = auto()
    Prophylaxis = auto()
    SingleDoseWhenSymptom = auto()
    FullTreatmentWhenSymptom = auto()
    SingleDoseParasiteDetect = auto()
    FullTreatmentParasiteDetect = auto()
    SingleDoseNewDetectionTech = auto()
    FullTreatmentNewDetectionTech = auto()


class AntipoliovirusDrug_Durability_Profile_Enum(Enum):
    FIXED_DURATION_CONSTANT_EFFECT = auto()
    CONCENTRATION_VERSUS_TIME = auto()


class BCGVaccine_Vaccine_Type_Enum(Enum):
    Generic = auto()
    TransmissionBlocking = auto()
    AcquisitionBlocking = auto()
    MortalityBlocking = auto()


class BitingRisk_Risk_Distribution_Enum(Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class BroadcastEventToOtherNodes_Node_Selection_Type_Enum(Enum):
    DISTANCE_ONLY = auto()
    MIGRATION_NODES_ONLY = auto()
    DISTANCE_AND_MIGRATION = auto()


class Contraceptive_Usage_Duration_Distribution_Enum(Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class ControlledVaccine_Vaccine_Type_Enum(Enum):
    Generic = auto()
    TransmissionBlocking = auto()
    AcquisitionBlocking = auto()
    MortalityBlocking = auto()


class DelayedIntervention_Delay_Period_Distribution_Enum(Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class DiagnosticTreatNeg_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class HIVDelayedIntervention_Delay_Period_Distribution_Enum(Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class HIVMuxer_Delay_Period_Distribution_Enum(Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class HealthSeekingBehaviorUpdateable_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class InterventionForCurrentPartners_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class InterventionForCurrentPartners_Prioritize_Partners_By_Enum(Enum):
    NO_PRIORITIZATION = auto()
    CHOSEN_AT_RANDOM = auto()
    LONGER_TIME_IN_RELATIONSHIP = auto()
    SHORTER_TIME_IN_RELATIONSHIP = auto()
    OLDER_AGE = auto()
    YOUNGER_AGE = auto()
    RELATIONSHIP_TYPE = auto()


class MDRDiagnostic_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class MalariaDiagnostic_Diagnostic_Type_Enum(Enum):
    Microscopy = auto()
    NewDetectionTech = auto()
    Other = auto()


class MalariaDiagnostic_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class MigrateIndividuals_Duration_At_Node_Distribution_Enum(Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class MigrateIndividuals_Duration_Before_Leaving_Distribution_Enum(Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class OutbreakIndividualTBorHIV_Infection_Type_Enum(Enum):
    HIV = auto()
    TB = auto()


class PolioVaccine_Vaccine_Type_Enum(Enum):
    tOPV = auto()
    bOPV = auto()
    mOPV_1 = auto()
    mOPV_2 = auto()
    mOPV_3 = auto()
    eIPV = auto()


class RTSSVaccine_Antibody_Type_Enum(Enum):
    CSP = auto()
    MSP1 = auto()
    PfEMP1_minor = auto()
    PfEMP1_major = auto()
    N_MALARIA_ANTIBODY_TYPES = auto()


class STIBarrier_Relationship_Type_Enum(Enum):
    TRANSITORY = auto()
    INFORMAL = auto()
    MARITAL = auto()
    COMMERCIAL = auto()
    COUNT = auto()


class STIIsPostDebut_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class SimpleBoosterVaccine_Vaccine_Type_Enum(Enum):
    Generic = auto()
    TransmissionBlocking = auto()
    AcquisitionBlocking = auto()
    MortalityBlocking = auto()


class SimpleDiagnostic_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class SimpleHealthSeekingBehavior_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class SimpleVaccine_Vaccine_Type_Enum(Enum):
    Generic = auto()
    TransmissionBlocking = auto()
    AcquisitionBlocking = auto()
    MortalityBlocking = auto()


class SmearDiagnostic_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class StiCoInfectionDiagnostic_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class TBHIVConfigurableTBdrug_Durability_Profile_Enum(Enum):
    FIXED_DURATION_CONSTANT_EFFECT = auto()
    CONCENTRATION_VERSUS_TIME = auto()


class TyphoidCarrierDiagnostic_Event_Or_Config_Enum(Enum):
    Config = auto()
    Event = auto()


class TyphoidVaccine_Mode_Enum(Enum):
    Shedding = auto()
    Dose = auto()
    Exposures = auto()


class UsageDependentBednet_Expiration_Period_Distribution_Enum(Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class Mated_Genetics_HEG_Enum(Enum):
    WILD = auto()
    HALF = auto()
    FULL = auto()
    NotMated = auto()


class Mated_Genetics_Pesticide_Resistance_Enum(Enum):
    WILD = auto()
    HALF = auto()
    FULL = auto()
    NotMated = auto()


class Released_Genetics_HEG_Enum(Enum):
    WILD = auto()
    HALF = auto()
    FULL = auto()
    NotMated = auto()


class Released_Genetics_Pesticide_Resistance_Enum(Enum):
    WILD = auto()
    HALF = auto()
    FULL = auto()
    NotMated = auto()


class ArtificialDiet_Artificial_Diet_Target_Enum(Enum):
    AD_WithinVillage = auto()
    AD_OutsideVillage = auto()


class BirthTriggeredIV_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class BirthTriggeredIV_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class InputEIR_Age_Dependence_Enum(Enum):
    OFF = auto()
    LINEAR = auto()
    SURFACE_AREA_DEPENDENT = auto()


class Larvicides_Habitat_Target_Enum(Enum):
    TEMPORARY_RAINFALL = auto()
    WATER_VEGETATION = auto()
    HUMAN_POPULATION = auto()
    CONSTANT = auto()
    BRACKISH_SWAMP = auto()
    MARSHY_STREAM = auto()
    LINEAR_SPLINE = auto()
    ALL_HABITATS = auto()


class MalariaChallenge_Challenge_Type_Enum(Enum):
    InfectiousBites = auto()
    Sporozoites = auto()


class MigrateFamily_Duration_At_Node_Distribution_Enum(Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class MigrateFamily_Duration_Before_Leaving_Distribution_Enum(Enum):
    CONSTANT_DISTRIBUTION = auto()
    UNIFORM_DISTRIBUTION = auto()
    GAUSSIAN_DISTRIBUTION = auto()
    EXPONENTIAL_DISTRIBUTION = auto()
    POISSON_DISTRIBUTION = auto()
    LOG_NORMAL_DISTRIBUTION = auto()
    DUAL_CONSTANT_DISTRIBUTION = auto()
    WEIBULL_DISTRIBUTION = auto()
    DUAL_EXPONENTIAL_DISTRIBUTION = auto()


class MosquitoRelease_Released_Gender_Enum(Enum):
    VECTOR_FEMALE = auto()
    VECTOR_MALE = auto()


class MosquitoRelease_Released_Sterility_Enum(Enum):
    VECTOR_FERTILE = auto()
    VECTOR_STERILE = auto()


class MosquitoRelease_Released_Wolbachia_Enum(Enum):
    WOLBACHIA_FREE = auto()
    VECTOR_WOLBACHIA_A = auto()
    VECTOR_WOLBACHIA_B = auto()
    VECTOR_WOLBACHIA_AB = auto()


class NLHTIVNode_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class NLHTIVNode_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class NodeLevelHealthTriggeredIV_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class NodeLevelHealthTriggeredIV_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class NodeLevelHealthTriggeredIVScaleUpSwitch_Demographic_Coverage_Time_Profile_Enum(
    Enum):
    Immediate = auto()
    Linear = auto()
    Sigmoid = auto()


class NodeLevelHealthTriggeredIVScaleUpSwitch_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class NodeLevelHealthTriggeredIVScaleUpSwitch_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class OvipositionTrap_Habitat_Target_Enum(Enum):
    TEMPORARY_RAINFALL = auto()
    WATER_VEGETATION = auto()
    HUMAN_POPULATION = auto()
    CONSTANT = auto()
    BRACKISH_SWAMP = auto()
    MARSHY_STREAM = auto()
    LINEAR_SPLINE = auto()
    ALL_HABITATS = auto()


class SpaceSpraying_Habitat_Target_Enum(Enum):
    TEMPORARY_RAINFALL = auto()
    WATER_VEGETATION = auto()
    HUMAN_POPULATION = auto()
    CONSTANT = auto()
    BRACKISH_SWAMP = auto()
    MARSHY_STREAM = auto()
    LINEAR_SPLINE = auto()
    ALL_HABITATS = auto()


class SpaceSpraying_Spray_Kill_Target_Enum(Enum):
    SpaceSpray_FemalesOnly = auto()
    SpaceSpray_MalesOnly = auto()
    SpaceSpray_FemalesAndMales = auto()
    SpaceSpray_Indoor = auto()


class TyphoidWASH_Mode_Enum(Enum):
    Shedding = auto()
    Dose = auto()
    Exposures = auto()


class NodeSetPolygon_Polygon_Format_Enum(Enum):
    SHAPE = auto()


class Action_Event_Type_Enum(Enum):
    INDIVIDUAL = auto()
    NODE = auto()
    COORDINATOR = auto()


class IncidenceCounter_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class IncidenceCounter_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class IncidenceCounterSurveillance_Counter_Event_Type_Enum(Enum):
    INDIVIDUAL = auto()
    NODE = auto()
    COORDINATOR = auto()


class IncidenceCounterSurveillance_Counter_Type_Enum(Enum):
    PERIODIC = auto()


class IncidenceCounterSurveillance_Target_Demographic_Enum(Enum):
    Everyone = auto()
    ExplicitAgeRanges = auto()
    ExplicitAgeRangesAndGender = auto()
    ExplicitGender = auto()
    PossibleMothers = auto()
    ArrivingAirTravellers = auto()
    DepartingAirTravellers = auto()
    ArrivingRoadTravellers = auto()
    DepartingRoadTravellers = auto()
    AllArrivingTravellers = auto()
    AllDepartingTravellers = auto()
    ExplicitDiseaseState = auto()


class IncidenceCounterSurveillance_Target_Gender_Enum(Enum):
    All = auto()
    Male = auto()
    Female = auto()


class LarvalHabitatMultiplierSpec_Habitat_Enum(Enum):
    TEMPORARY_RAINFALL = auto()
    WATER_VEGETATION = auto()
    HUMAN_POPULATION = auto()
    CONSTANT = auto()
    BRACKISH_SWAMP = auto()
    MARSHY_STREAM = auto()
    LINEAR_SPLINE = auto()
    ALL_HABITATS = auto()


class ResistanceHegGenetics_HEG_Enum(Enum):
    WILD = auto()
    HALF = auto()
    FULL = auto()
    NotMated = auto()


class ResistanceHegGenetics_Pesticide_Resistance_Enum(Enum):
    WILD = auto()
    HALF = auto()
    FULL = auto()
    NotMated = auto()


class Responder_Threshold_Type_Enum(Enum):
    COUNT = auto()
    PERCENTAGE = auto()
    PERCENTAGE_EVENTS = auto()


class ResponderSurveillance_Threshold_Type_Enum(Enum):
    COUNT = auto()
    PERCENTAGE = auto()
    PERCENTAGE_EVENTS = auto()
