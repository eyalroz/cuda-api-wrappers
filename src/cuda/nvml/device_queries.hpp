/**
 * @file
 *
 * @brief Contains the API wrapper for NVL library D
 *
 * @note This file does not include wrappers for
 * `nvmlSystemGetCudaDriverVersion()`,
 * since the CUDA driver API itself can be used for this functionality; see
 * @ref cuda::version_numbers::driver() .
 */

#include "types.hpp"
#include <chrono>
#include <string>

namespace cuda {

namespace nvml {

nvmlEnableState_t api_restriction (device_t device, nvmlRestrictedAPI_t apiType);
unsigned int adaptive_clock_info_status (device_t device );

unsigned int ApplicationsClock (device_t device, nvmlClockType_t clockType,  );
nvmlDeviceArchitecture_t Architecture (device_t device);
nvmlDeviceAttributes_t Attributes_v2 (device_t device);

// currently enabled state, default enabled state
::std::pair<nvmlEnableState_t, nvmlEnableState_t> AutoBoostedClocksEnabled (device_t device);

nvmlBAR1Memory_t BAR1MemoryInfo (device_t device);
unsigned int BoardId (device_t device)
::std::string BoardPartNumber (device_t device);
nvmlBrandType_t Brand (device_t device);
nvmlBridgeChipHierarchy_t BridgeChipInfo (device_t device);
unsigned int Clock (device_t device, nvmlClockType_t clockType, nvmlClockId_t clockId)
unsigned int ClockInfo (device_t device, nvmlClockType_t type)
nvmlComputeMode_t ComputeMode (device_t device);
::std::vector<nvmlProcessInfo_t> ComputeRunningProcesses (device_t device);
int CudaComputeCapability (device_t device, int* major);
unsigned int CurrPcieLinkGeneration (device_t device)
unsigned int CurrPcieLinkWidth (device_t device)
unsigned long long CurrentClocksThrottleReasons (device_t device)

struct utilization_info_t {
    unsigned int utilization;
    ::std::chrono::microseconds sampling_period;
};

utilization_info_t DecoderUtilization (device_t device);

unsigned int DefaultApplicationsClock (device_t device, nvmlClockType_t clockType)
nvmlEnableState_t DefaultEccMode (device_t device);
nvmlEccErrorCounts_t DetailedEccErrors (nvmlEccCounterType_t counterType);
nvmlEnableState_t DisplayActive (device_t device);
nvmlEnableState_t DisplayMode (device_t device);

template <typename T>
struct current_and_pending_t{
    T current;
    T pending;
};

current_and_pending_t<nvmlDriverModel_t> DriverModel (device_t device);
current_and_pending_t<nvmlEnableState_t> EccMode (device_t device);
unsigned int EncoderCapacity (device_t device, nvmlEncoderType_t encoderQueryType);
::std::vector<nvmlEncoderSessionInfo_t> EncoderSessions (device_t device);

struct encoder_statistics_t {
    unsigned int sessionCount;
    unsigned int averageFps;
    unsigned int averageLatency;
};

encoder_statistics_t EncoderStats (device_t device);
utilization_info_t EncoderUtilization (device_t device);
unsigned int EnforcedPowerLimit (device_t device)
::std::vector<nvmlFBCSessionInfo_t> FBCSessions (device_t device);
nvmlFBCStats_t FBCStats (device_t device);
nvmlFanControlPolicy_t FanControlPolicy_v2 (unsigned int fan);
unsigned int FanSpeed (device_t device);
unsigned int FanSpeed_v2 (device_t device, unsigned int  fan);
unsigned int GpuMaxPcieLinkGeneration (nvmlDevice_t device);
current_and_pending_t<nvmlGpuOperationMode_t> GpuOperationMode (device_t device);
::std::vector<nvmlProcessInfo_t> GraphicsRunningProcesses (device_t device);
nvmlDevice_t HandleByIndex_v2 (unsigned int  index);
nvmlDevice_t HandleByPciBusId_v2 (const char* pciBusId);
device_t HandleBySerial (const char* serial);
device_t HandleByUUID ( const char GpuMaxPcieLinkGeneration);
unsigned int Index (device_t device);
unsigned int InforomConfigurationChecksum (device_t device);
void InforomImageVersion (device_t device, char* version, unsigned int  length );
void InforomVersion (device_t device, nvmlInforomObject_t object, char* version, unsigned int  length );
unsigned int IrqNum (device_t device);
::std::vector<nvmlProcessInfo_t> MPSComputeRunningProcesses_v3 (device_t device);
unsigned int MaxClockInfo (device_t device, nvmlClockType_t type);
unsigned int MaxCustomerBoostClock (device_t device, nvmlClockType_t clockType);
unsigned int MaxPcieLinkGeneration (device_t device);
unsigned int MaxPcieLinkWidth (device_t device);
unsigned int MemoryBusWidth (device_t device);
unsigned long long MemoryErrorCounter (device_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType);
nvmlMemory_t MemoryInfo (device_t device);
unsigned int MinMaxFanSpeed (device_t device, unsigned int* minSpeed);
unsigned int MinorNumber (device_t device);
unsigned int MultiGpuBoard (device_t device);
void Name (device_t device, char* name, unsigned int  length );
unsigned int NumFans (device_t device);
unsigned int NumGpuCores (device_t device);
nvmlGpuP2PStatus_t P2PStatus (nvmlGpuP2PCapsIndex_t p2pIndex);
nvmlPciInfo_t PciInfo_v3 (device_t device);
unsigned int PcieLinkMaxSpeed (device_t device);
unsigned int PcieReplayCounter (device_t device);
unsigned int PcieSpeed (device_t device);
unsigned int PcieThroughput (device_t device, nvmlPcieUtilCounter_t counter);
nvmlPstates_t PerformanceState (device_t device);
nvmlEnableState_t PersistenceMode (device_t device);
unsigned int PowerManagementDefaultLimit (device_t device);
unsigned int PowerManagementLimit (device_t device);

struct limit_pair_t {
    unsigned int min, max;
};

limit_pair_t PowerManagementLimitConstraints (device_t device);
nvmlEnableState_t PowerManagementMode (device_t device);
nvmlPowerSource_t PowerSource (device_t device);
nvmlPstates_t PowerState (device_t device);
unsigned int PowerUsage (device_t device);

struct remapped_rows_info_t {
    unsigned int corrRows;
    unsigned int uncRows;
    unsigned int isPending;
    unsigned int failureOccurred;
};

remapped_rows_info_t RemappedRows (device_t device);

struct page_retirement_info_t {
    nvmlPageRetirementCause_t cause;
    unsigned int pageCount;
    unsigned long long addresses,
    unsigned long long timestamps);
};
page_retirement_info_t RetiredPages (device_t device);
nvmlEnableState_t RetiredPagesPendingStatus (device_t device);
nvmlRowRemapperHistogramValues_t RowRemapperHistogram (device_t device);
::std::vector<nvmlSample_t> Samples (device_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType);
void Serial (device_t device, char* serial, unsigned int  length );
unsigned long long SupportedClocksThrottleReasons (device_t device);

// TODO: Fix me
using mhz = unsigned;
// MHz
::std::vector<mhz> SupportedGraphicsClocks (device_t device, mhz memory_clock);

// MHz
::std::vector<mhz> SupportedMemoryClocks (device_t device);

unsigned int TargetFanSpeed (device_t device, unsigned int fan);
unsigned int Temperature (device_t device, nvmlTemperatureSensors_t sensorType);
unsigned int TemperatureThreshold (device_t device, nvmlTemperatureThresholds_t thresholdType);
nvmlGpuThermalSettings_t ThermalSettings (unsigned int  sensorIndex);
nvmlGpuTopologyLevel_t TopologyCommonAncestor (device_t device2);
::std::vector<device_t> TopologyNearestGpus (device_t device, nvmlGpuTopologyLevel_t level);
unsigned long long TotalEccErrors (device_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType);
unsigned long long TotalEnergyConsumption (device_t device);
void UUID (device_t device, char* uuid, unsigned int  length );
nvmlUtilization_t UtilizationRates (device_t device);
void VbiosVersion (device_t device, char* version, unsigned int  length );
nvmlViolationTime_t ViolationStatus (nvmlPerfPolicyType_t perfPolicyType);
nvmlReturn_t nvmlDeviceOnSameBoard (device_t device1, device_t device2, int* onSameBoard );
nvmlReturn_t nvmlDeviceResetApplicationsClocks (device_t device );
nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled (device_t device, nvmlEnableState_t enabled );
nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled (device_t device, nvmlEnableState_t enabled, unsigned int  flags );
nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2 (device_t device, unsigned int  fan );
nvmlReturn_t nvmlDeviceSetFanControlPolicy (device_t device, unsigned int  fan, nvmlFanControlPolicy_t policy );
nvmlReturn_t nvmlDeviceSetTemperatureThreshold (device_t device, nvmlTemperatureThresholds_t thresholdType, int* temp );
nvmlReturn_t nvmlDeviceValidateInforom (device_t device );
nvmlReturn_t nvmlSystemGetTopologyGpuSet ( unsigned int  cpuNumber, unsigned int* count, nvmlDevice_t* deviceArray );
nvmlReturn_t nvmlVgpuInstanceGetMdevUUID ( nvmlVgpuInstance_t vgpuInstance, char* mdevUuid, unsigned int  size );

} // namespace nvml

} // namespace cuda
