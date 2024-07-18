# MultimodalMedPOCs

This repo holds some POCs developed for the quantification of Parkinson's states, but it could be more broadly used for other conditions. Each directory is dedicated to a specific POC and has its own requirements. For all POCs, except for FruitNinja, openly available data was used, i.e., Youtube videos of Parkinson patients. However, the data is not provided in the git.

- [x] Finger Tapping - Based on one of the tests performed during Parkinson's evaluation. MediaPipe is used for effective finger tracking. Run setup.sh for requirements installation and for the model to download into the correct location. Data needs to be stored under 'data'.
- [x] Speech - Based on research of changes in vocal and speech characteristics of Parkinson's patients. OpenAI is used for transcriptions, which requires owning an API key.
- [ ] FruitNinja - Exploration of changes in interaction with the phone during different states. Waits to be uploaded after the necessary cleanup.
