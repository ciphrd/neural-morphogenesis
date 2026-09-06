import type { RunSettings } from "../gpu/types"
import { defaultChemicalChannelProfiles } from "../gpu/chemicalChannels"
import sharedDefaultRunSettingsConfig from "../../../core/config.json";
const sharedDefaultRunSettings = sharedDefaultRunSettingsConfig.run;
const chemistry = sharedDefaultRunSettingsConfig.chemistry;

// Offline/first-visit configuration. This JSON is part of the trainer's
// canonical configuration too, so keeping the random-brain playground usable
// does not require a second set of values maintained in TypeScript.
export const DEFAULT_RUN_SETTINGS = {
  ...sharedDefaultRunSettings,
  baseResolution: chemistry.baseResolution,
  chemicalCommunicationArchitecture: chemistry.chemicalCommunicationArchitecture,
  decay: chemistry.decay,
  depositRate: chemistry.depositRate,
  normalizeDepositsByLocalDensity: chemistry.normalizeDepositsByLocalDensity,
  maxEnvWrite: chemistry.maxEnvWrite,
  chemicalValueInputMultiplier: chemistry.chemicalValueInputMultiplier,
  chemicalGradientInputScale: chemistry.chemicalGradientInputScale,
  communicationSpeed: chemistry.communicationSpeed,
  neuralUpdatesPerMacro: chemistry.neuralUpdatesPerMacro,
  initialConditionChannel: chemistry.initialConditionChannel,
  channels: chemistry.channels.length,
  chemicalChannelProfiles: defaultChemicalChannelProfiles(chemistry.channels.length),
} as RunSettings

/** Returns canonical defaults until a live server supplies run settings. */
export function loadInitialRunSettings(): RunSettings {
  return DEFAULT_RUN_SETTINGS
}
