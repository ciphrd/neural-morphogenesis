import type { RunSettings } from "../gpu/types"
import { defaultChemicalChannelProfiles } from "../gpu/chemicalChannels"
import sharedDefaultRunSettingsConfig from "../../../core/config.json";
const sharedDefaultRunSettings = sharedDefaultRunSettingsConfig.run;

// Offline/first-visit configuration. This JSON is part of the trainer's
// canonical configuration too, so keeping the random-brain playground usable
// does not require a second set of values maintained in TypeScript.
export const DEFAULT_RUN_SETTINGS = {
  ...sharedDefaultRunSettings,
  chemicalChannelProfiles: defaultChemicalChannelProfiles(sharedDefaultRunSettings.channels),
} as RunSettings

/** Returns canonical defaults until a live server supplies run settings. */
export function loadInitialRunSettings(): RunSettings {
  return DEFAULT_RUN_SETTINGS
}
