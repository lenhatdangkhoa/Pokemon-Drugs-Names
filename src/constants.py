"""
Constants for Pokemon adversarial hallucination experiment.
"""

# Experiment conditions - only 6 required:
# Drug Dosing Prompt: default, mitigation, temp0
# Drug Indication Prompt: medication_indication, medication_indication_mitigation, medication_indication_temp0
CONDITIONS = [
    "default",                          # Drug Dosing Prompt - Default
    "mitigation",                       # Drug Dosing Prompt - Default + Mitigation
    "temp0",                            # Drug Dosing Prompt - Temp 0
    "medication_indication",            # Drug Indication Prompt - Default
    "medication_indication_mitigation", # Drug Indication Prompt - Default + Mitigation
    "medication_indication_temp0"       # Drug Indication Prompt - Temp 0
]
