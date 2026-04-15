This repository contains the Python codebase that accompanies the following article:  
Péter Lénárt<sup>1,2,3</sup>, Viktor Háda<sup>3</sup>, Kinga Komka<sup>4</sup>, \*Gergely Tóth<sup>5</sup>, \*Gitta Schlosser<sup>1</sup>
[**Deterministic Acceptance Limits for Statistical Equivalence Testing in Hydrogen/Deuterium Exchange Mass Spectrometry**]
*Journal of the American Society for Mass Spectrometry*, 2026, (https://doi.org/10.1021/jasms.6c00035)

---

## Setup
The codebase currently offers Windows compatibility and was tested with Microsoft Windows [Version 10.0.26200.8039].

1. Download the latest version of Python from <https://www.python.org/downloads>. We recommend using the standalone installer. During installation, check **"Add python.exe to PATH"**. The development was done using Python version 3.8.16 and verified compatible with Python 3.14.3.
    <img width="752" height="332" alt="Picture1" src="https://github.com/user-attachments/assets/e683e5d4-81e3-490b-9588-3c791f21d502" />


2. Download the repository by clicking the green "**Code**" button and selecting "**Download ZIP**". Extract the downloaded `.zip` folder and move its contents into a folder that will be used for evaluation, e.g.:

   ```
   C:\Users\YourName\HDX-MS_Evaluation
   ```

3. Type `cmd` in the Windows search bar and open **Command Prompt**.

    <img width="507" height="291" alt="Picture3" src="https://github.com/user-attachments/assets/7ab57c5a-32ce-4ce5-b481-f77136eb0243" />


4. Inside the Command Prompt, navigate to the folder that contains the codebase using this command:

   ```cmd
   cd C:\Users\YourName\HDX-MS_Evaluation
   ```
   Replace the example path with your own folder path.
   
    <img width="752" height="189" alt="Picture4" src="https://github.com/user-attachments/assets/33aab552-c65f-4317-be95-4b96d52f6f56" />

   Notice that the file path displayed in the Command Prompt changes to the folder you switched into.

5. Create a new virtual environment using the following command:

   ```cmd
   python -m venv hdx_ms_virtual_env
   ```
   
   This can take a few seconds. You can choose an alternative name by replacing `hdx_ms_virtual_env` with your preferred name.

6. Activate the virtual environment:

   ```cmd
   .\hdx_ms_virtual_env\Scripts\activate
   ```
    <img width="752" height="60" alt="Picture6" src="https://github.com/user-attachments/assets/d53194dd-e691-4d48-8881-690b15ced1bb" />

   Note that the name of the activated environment appears in parentheses before the current file path in the console.

7. Upgrade pip, which is used to install the required Python packages:

   ```cmd
   python -m pip install --upgrade pip
   ```
   You will see something similar in console:
    <img width="752" height="147" alt="Picture7" src="https://github.com/user-attachments/assets/3b7d8818-837f-4119-993b-0c4be3e46af2" />

8. Install the required packages:

   ```cmd
   pip install -r requirements.txt
   ```

   You will see a long list of packages being downloaded and installed. This process will take a few seconds.
   Development was done using: pandas v1.1.3, numpy v1.24.3, scipy v1.5.4, matplotlib v3.2.2, statsmodels v0.14.1,
   openpyxl v3.1.5, fpdf2 v2.8.3, tqdm v4.67.1

---

## Evaluation

### Step 1: Prepare your data folders

Create a `Data` folder inside your evaluation folder. In the above example the evaluation folder was named “HDX-MS_Evaluation”. Inside the Data folder create at least these two folders:

<img width="601" height="221" alt="Picture8" src="https://github.com/user-attachments/assets/8df20038-2f55-494b-8863-b78a63e9ca16" />


Inside the Protein_to_be_evaluated_1 folder create at least these two folders:
<img width="607" height="232" alt="Picture9" src="https://github.com/user-attachments/assets/6677a32b-0337-4f1e-a5b3-bc06953fc383" />

So inside the `Data` folder the data files will be organized using the following structure:

```
Data/
├── Protein_to_be_evaluated_1/
   ├── Null_experiment_1/
   │   └── null_experiment_data.csv
   ├── Reference/
   │   └── reference_data.csv
   ├── Biosimilar_candidate_1/
       └── candidate_1_data.csv
```

- The **Null_experiment** folder contains the `.csv` file with null experiment data, e.g., HDX-MS data for all peptide-labeling timepoints for 8 replicates of Remicade.
- The **Reference** folder contains the `.csv` file with reference data, e.g., HDX-MS data for all peptide-labeling timepoints for 3 replicates of Remicade.
- Each **Biosimilar_candidate** folder contains the `.csv` file with data for a protein sample to compare against the reference, e.g., HDX-MS data for all peptide-labeling timepoints for 3 replicates of Renflexis.

You can add more biosimilar candidates by creating additional numbered folders, e.g., `Biosimilar_candidate_2`, `Biosimilar_candidate_3`, etc. inside a protein folder.  
<img width="632" height="302" alt="Picture10" src="https://github.com/user-attachments/assets/097f039f-f797-4fd7-b695-3d15f66fd779" />

You can also evaluate multiple proteins by adding more `Protein_to_be_evaluated_N` folders to the `Data` directory.
<img width="645" height="298" alt="Picture11" src="https://github.com/user-attachments/assets/a16d64e6-4424-43ed-a835-7ecfa6a308e4" />

### Step 2: CSV format

The `.csv` file for the **null experiment** data should have the following columns:  
Sequence, HX time, Uptake (Da), Replicate number

  <img width="647" height="460" alt="Picture12" src="https://github.com/user-attachments/assets/a081b3e3-0100-46c0-8565-8be802997479" />  

  > **Note:** The exact column names shown above must be present and are **case sensitive**.  

In this example 8 replicates were used for the **null experiment**, but you can use as many as you wish.

The `.csv` file for the **reference** or **biosimilar candidate** data should have the following columns:  
Protein state, Sequence, HX time, Uptake (Da), Uptake SD (Da), Replicate number

  <img width="752" height="339" alt="Picture13" src="https://github.com/user-attachments/assets/8aaea2cf-207e-46e8-94fc-c414333d20e9" />  
  
  > **Note:** The exact column names shown above must be present and are **case sensitive**.  

In this example 3 replicates were used for the **reference** and **biosimilar candidate** data, but you can use as many as you wish.

### Step 3: Configure the evaluation

If you have not already done so, navigate to the codebase folder and activate the virtual environment, see Setup steps 4 and 6.

Open the `config.txt` file, which contains all evaluation parameters and descriptions for each method.  
Most importantly, you need to:
1. Indicate how many proteins you wish to evaluate.
2. Set the paths to your null experiment, reference, and candidate `.csv` files.

For example, to evaluate two proteins - Infliximab with 3 biosimilar candidates and NIST mAb with 1 biosimilar candidate - configure the paths under `[PROTEIN_TO_EVALUATE_1]` and `[PROTEIN_TO_EVALUATE_2]`. To add a third protein, add a `[PROTEIN_TO_EVALUATE_3]` section and set the corresponding file paths.

<img width="752" height="277" alt="Picture14" src="https://github.com/user-attachments/assets/4658ddbe-2a40-4d2d-b198-b19f32dcb740" />

Set the methods you wish to use to `true`:

```ini
run_complete_enumeration = true
run_monte_carlo = true
run_direct_percentile = true
run_resampling = true
run_partitioned_limits = true
```

If you want to generate the limits without running the evaluation, set `run_evaluation = false`. To both calculate the limits and evaluate the sample, set `run_evaluation = true`.  
Note that if `run_evaluation = true` but no method above is set to `true`, you will get an error.

### Step 4: Run the analysis

```cmd
python main.py
```

After the analysis finishes, it will create an `outputs` folder, if it does not already exist, containing a subfolder named with the format `run_YYYY-MM-DD_HH-MM-SS`, so each analysis run is timestamped. If you want to repeat the analysis after changing the config settings just run the command again. Since the output is timestamped you don't have to worry about overwriting results upon repeated analysis.

### Step 5: Deactivate the environment

When you are finished working, deactivate the virtual environment:

```cmd
deactivate
```

---

## Commands to run if you closed and reopened the console, but the setup was already done and the data files are in place

Run these commands in sequence:

```cmd
cd C:\Users\YourName\HDX-MS_Evaluation
.\hdx_ms_virtual_env\Scripts\activate
python main.py
```
<img width="1278" height="482" alt="Picture16" src="https://github.com/user-attachments/assets/2390b8c3-1354-4519-a57f-1df058aca7a7" />

---

## Citation and license

If you use this software or a modified version of it, please cite the publication.

The algorithm is provided under the Apache 2.0 license. See the `LICENSE.txt` file for details.

---

## Troubleshooting

- If a module is reported as missing, install it by running:

  ```cmd
  pip install <module_name>
  ```

- If you want to remove the virtual environment, navigate to the folder where you created it and run:

  ```cmd
  rmdir /s /q hdx_ms_virtual_env
  ```
Use the name of the virtual environment instead of hdx_ms_virtual_env if you chose a different name previously.
