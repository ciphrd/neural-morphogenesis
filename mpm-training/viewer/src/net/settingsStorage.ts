import type { RunSettings } from "../gpu/types"
import { defaultChemicalChannelProfiles } from "../gpu/chemicalChannels"
import sharedDefaultRunSettings from "../../../core/default_run_settings.json"

// Offline/first-visit configuration. This JSON is part of the trainer's
// canonical configuration too, so keeping the random-brain playground usable
// does not require a second set of values maintained in TypeScript.
export const DEFAULT_RUN_SETTINGS: RunSettings = {
  ...(sharedDefaultRunSettings as RunSettings),
  chemicalChannelProfiles: defaultChemicalChannelProfiles(sharedDefaultRunSettings.channels),
}

/** Returns canonical defaults until a live server supplies run settings. */
export function loadInitialRunSettings(): RunSettings {
  return DEFAULT_RUN_SETTINGS
}
