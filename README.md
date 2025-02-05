## Cerberus-NIDS

This project provides a quick test for validating our proposed methods, aimed at evaluating the security and privacy protection capabilities of intelligent network intrusion detection models. Below is an overview of the project structure and files:

## File Structure Description

### 1. `data_example/pcap_data.json`

This file is a sample of the victim’s private training data, including:

- Benign traffic and six malicious categories.
- Data has been cleaned and anonymized to meet relevant standards.

### 2. `data_example/checkpoints`

- This folder contains the trained victim model.
- Includes the autoencoder model used in our proposed algorithm.

### 3. `data_example/JBDA_query_example` and `data_example/knockoff_query_example`

- These folders store examples of attack queries:
	- `JBDA_query_example`: Examples of queries based on JBDA attacks.
	- `knockoff_query_example`: Examples of queries based on knockoff attacks.

### 4. `utils/utils.py`

- This script includes utility functions and tools for data processing.

### 5. `utils/victim_utils.py`

- Contains the training and testing process of the victim model.

### 6. `utils/attack_utils.py`

- Implements the simulation process of attackers' attacks.

### 7. `utils/defend_utils.py`

- Provides a quick example of our proposed defense algorithm.
- Includes implementations of comparative defense algorithms.

## Usage Instructions

1. Prepare Data: Ensure that `data_example/pcap_data.json` and the related model files are correctly extracted and placed in the specified paths.
2. Train and Test Models: Use `utils/victim_utils.py` to train and validate the victim model.
3. Simulate Attacks: Use `utils/attack_utils.py` to test attacks based on `JBDA_query_example` and `knockoff_query_example`.
4. Test Defenses: Run the proposed defense algorithm through `utils/defend_utils.py` and compare its performance with other defense methods.

## Notes

- All data has been anonymized to meet privacy protection standards.

## Contact

If you have any questions or suggestions, please contact the project author for support.







