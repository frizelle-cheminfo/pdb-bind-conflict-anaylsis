# Cross-Database Binding Affinity Conflicts: Implications for Machine Learning Model Training

**Authors:** [Author names to be added]

**Affiliation:** [Affiliations to be added]

**Corresponding Author:** [Contact information to be added]

---

## Abstract

Accurate binding affinity data is essential for training machine learning models in drug discovery, yet the reliability of publicly available databases remains understudied. We performed a systematic cross-database comparison of binding affinity values between PDBbind (v2020), BindingDB, and ChEMBL for identical protein-ligand complexes. Among 3,910 measurement type-matched comparisons (comparing Ki to Ki, Kd to Kd, or IC50 to IC50), we found that **25.7% showed conflicts exceeding 10-fold differences** in reported affinity values. This conflict rate was remarkably consistent across measurement types: 24.9% for IC50, 25.7% for Kd, and 27.4% for Ki comparisons. In contrast, comparisons between mismatched measurement types (e.g., Ki vs IC50) showed conflict rates of 42-56%, highlighting the importance of measurement type filtering. We further demonstrate that machine learning models trained on PDBbind data show significant performance degradation when evaluated with scaffold-based cross-validation (R^2^ = 0.41) compared to random splits (R^2^ = 0.52), suggesting overfitting to chemical series. These findings have important implications for computational drug discovery: researchers should (1) filter training data by measurement type, (2) validate extreme affinity values across databases, and (3) use scaffold-based splits to estimate real-world predictive performance.

**Keywords:** binding affinity, machine learning, drug discovery, PDBbind, BindingDB, data quality, cross-validation

---

## 1. Introduction

The application of machine learning (ML) to predict protein-ligand binding affinity has emerged as a promising approach in computational drug discovery (Yang et al., 2019; Rifaioglu et al., 2019; Jimenez-Luna et al., 2020). These models are typically trained on curated databases of experimentally determined binding affinities, with PDBbind being the most widely used resource that links structural data from the Protein Data Bank to quantitative binding measurements (Wang et al., 2004). However, the quality and consistency of the underlying experimental data directly impacts model performance and reliability.

Binding affinity is typically reported as equilibrium dissociation constants (Kd or Ki) or half-maximal inhibitory concentrations (IC50). While Ki and Kd represent thermodynamic equilibrium constants that are intrinsic properties of the protein-ligand interaction, IC50 values are assay-dependent and can vary significantly based on experimental conditions (Kalliokoski et al., 2013). The Cheng-Prusoff equation describes the relationship between IC50 and Ki, demonstrating that IC50 values are typically 2-10 times higher than corresponding Ki values depending on substrate concentration (Cheng & Prusoff, 1973).

Despite the widespread use of binding affinity databases for ML model training, systematic comparisons of affinity values across databases remain limited. Previous studies have focused primarily on individual database curation (Liu et al., 2007; Gaulton et al., 2017) rather than cross-database validation. Understanding the extent of discrepancies between databases is crucial for:

1. Assessing the reliability of training data
2. Identifying potential sources of prediction errors
3. Developing best practices for data preprocessing

In this study, we perform the first comprehensive cross-database comparison of binding affinity values between PDBbind, BindingDB, and ChEMBL. We systematically match protein-ligand pairs using canonical SMILES representations and quantify the prevalence and magnitude of affinity conflicts. We further investigate how these data quality issues, combined with common evaluation practices, may lead to overestimation of ML model performance.

---

## 2. Methods

### 2.1 Data Sources

**PDBbind v2020** (Wang et al., 2004): We obtained binding affinity data for 19,443 protein-ligand complexes from the PDBbind database (version 2020, general set). Affinity values were converted to pKd units using the formula: pKd = -log10(affinity_M), where affinity_M is the binding constant in molar units. Ligand structures were extracted from PDB coordinate files and converted to canonical SMILES using RDKit (version 2023.09) (Landrum, 2023).

**BindingDB** (Gilson et al., 2016): Cross-database matches were identified by querying the BindingDB REST API using PDB identifiers. For each PDB structure in PDBbind, we retrieved all associated binding affinity measurements and matched ligands using canonical SMILES comparison.

**ChEMBL** (Mendez et al., 2019): Additional binding data was retrieved from ChEMBL version 33 using the ChEMBL web services API, matching compounds by canonical SMILES representations.

### 2.2 Ligand Matching

Ligands were matched between databases using canonical SMILES representations generated by RDKit. This approach ensures that tautomeric and stereochemical variations are handled consistently. Only exact SMILES matches were considered to minimize false positive matches.

### 2.3 Conflict Definition

Following established conventions in the field (Kramer et al., 2012), we defined a conflict as a difference of greater than 1.0 pKd units between databases, corresponding to a >10-fold difference in binding affinity. This threshold accounts for typical experimental variability in binding assays while identifying genuinely discrepant values.

Conflicts were calculated as:
```
delta_pKd = pKd_database2 - pKd_database1
conflict = |delta_pKd| > 1.0
```

### 2.4 Measurement Type Classification

Affinity measurements were classified by type: Ki (inhibition constant), Kd (dissociation constant), IC50 (half-maximal inhibitory concentration), or Unknown (when not specified). Comparisons were stratified into:
- **Type-matched (valid)**: Same measurement type in both databases
- **Type-mismatched (invalid)**: Different measurement types

Only type-matched comparisons were considered valid for conflict rate calculation, as comparing IC50 to Ki/Kd values is methodologically inappropriate due to their different biochemical meanings (Kalliokoski et al., 2013).

### 2.5 Machine Learning Models

To assess the impact of data quality on predictive modeling, we trained three ML models using 10 RDKit molecular descriptors:
- Molecular weight (MolWt)
- Lipophilicity (LogP)
- Topological polar surface area (TPSA)
- Hydrogen bond donors/acceptors
- Rotatable bonds
- Aromatic rings
- Fraction sp3 carbons
- Heavy atom count
- Ring count

Models were evaluated using two cross-validation strategies:
1. **Random 5-fold CV**: Standard random splitting of data
2. **Scaffold 5-fold CV**: Splitting by Murcko scaffolds to ensure chemically distinct compounds in train/test sets (Bemis & Murcko, 1996)

---

## 3. Results

### 3.1 Cross-Database Matching

We successfully matched 5,438 PDBbind entries to BindingDB records based on identical canonical SMILES representations. Of these, 3,910 comparisons (71.9%) involved matching measurement types and were considered valid for conflict analysis.

### 3.2 Conflict Rates by Measurement Type

**Table 1.** Conflict rates for type-matched cross-database comparisons.

| Measurement Type | N Comparisons | N Conflicts | Conflict Rate |
|-----------------|---------------|-------------|---------------|
| IC50 vs IC50 | 2,122 | 529 | 24.9% |
| Kd vs Kd | 913 | 235 | 25.7% |
| Ki vs Ki | 875 | 240 | 27.4% |
| **Total (type-matched)** | **3,910** | **1,004** | **25.7%** |

The conflict rate was remarkably consistent across measurement types, ranging from 24.9% to 27.4%. This suggests that data quality issues are not specific to any particular assay type but represent a fundamental challenge in binding affinity measurements.

### 3.3 Impact of Measurement Type Matching

**Table 2.** Conflict rates for type-mismatched comparisons.

| PDBbind Type | BindingDB Type | N | Conflict Rate |
|-------------|----------------|---|---------------|
| IC50 | Kd | 296 | 54.7% |
| Kd | IC50 | 283 | 42.8% |
| IC50 | Ki | 248 | 49.6% |
| Kd | Ki | 177 | 56.5% |
| Ki | Kd | 171 | 54.4% |
| Ki | IC50 | 162 | 51.9% |

Type-mismatched comparisons showed dramatically higher conflict rates (42-57%) compared to type-matched comparisons (25-27%). This finding emphasizes the critical importance of filtering training data by measurement type before model development, consistent with recommendations from Kalliokoski et al. (2013).

### 3.4 Distribution of Affinity Differences

![Figure 3. Distribution of pKd differences between PDBbind and BindingDB](figures/Figure_3_delta_distribution.png)

**Figure 3.** Distribution of pKd differences (BindingDB - PDBbind) for type-matched comparisons. The shaded region indicates values within the 10-fold agreement threshold (|delta_pKd| < 1.0). The distribution shows a slight positive skew, suggesting BindingDB values tend to be marginally higher than PDBbind values.

### 3.5 Correlation Between Databases

![Figure 2. Correlation between PDBbind and BindingDB values](figures/Figure_2_correlation.png)

**Figure 2.** Scatter plot of PDBbind pKd values versus BindingDB pKd values for type-matched comparisons. The diagonal line represents perfect agreement. While strong overall correlation is observed (Pearson r = 0.85), substantial scatter exists, particularly at intermediate affinity values (pKd 5-8).

### 3.6 Conflict Overview by Protein Class

![Figure 1. Cross-database conflict overview](figures/Figure_1_conflict_overview.png)

**Figure 1.** Overview of cross-database conflicts by protein class. Kinases and proteases, which are heavily represented in drug discovery datasets, show conflict rates consistent with the overall average of 25.7%.

### 3.7 Machine Learning Model Performance

**Table 3.** Machine learning model performance with different cross-validation strategies.

| Model | Split Type | R^2^ (mean +/- SD) | RMSE (mean +/- SD) |
|-------|-----------|------------------|-------------------|
| XGBoost | Random | 0.52 +/- 0.003 | 1.30 +/- 0.01 |
| LightGBM | Random | 0.50 +/- 0.007 | 1.33 +/- 0.01 |
| Random Forest | Random | 0.48 +/- 0.004 | 1.36 +/- 0.01 |
| XGBoost | Scaffold | 0.41 +/- 0.014 | 1.44 +/- 0.02 |
| LightGBM | Scaffold | 0.39 +/- 0.012 | 1.46 +/- 0.03 |
| Random Forest | Scaffold | 0.37 +/- 0.015 | 1.48 +/- 0.02 |

All models showed substantial performance degradation when evaluated with scaffold-based cross-validation compared to random splits. The R^2^ dropped by approximately 0.11 units (21% relative decrease), and RMSE increased by approximately 0.11 pKd units. This suggests that random split evaluations may overestimate real-world predictive performance due to information leakage between chemically similar compounds, as previously noted by Brown et al. (2009).

---

## 4. Discussion

### 4.1 Prevalence of Cross-Database Conflicts

Our finding that 25.7% of type-matched cross-database comparisons show >10-fold affinity differences has significant implications for ML model training. This level of conflict cannot be attributed solely to experimental variability, which is typically 2-3 fold for well-controlled binding assays (Kramer et al., 2012; Brown et al., 2009). Potential sources of these discrepancies include:

1. **Assay condition differences**: Temperature, buffer composition, and protein construct variations
2. **Curation errors**: Transcription mistakes during manual database curation
3. **Unit conversion errors**: Confusion between nM, uM, and mM units
4. **Stereochemistry issues**: Different stereoisomers reported under the same identifier

### 4.2 Importance of Measurement Type Matching

The dramatically higher conflict rates observed for type-mismatched comparisons (42-57% vs 25-27%) validate the theoretical expectations from the Cheng-Prusoff equation (Cheng & Prusoff, 1973). Researchers often pool Ki, Kd, and IC50 values together when training ML models, assuming they are roughly equivalent. Our data strongly argues against this practice.

**Recommendation 1**: Filter training data to include only one measurement type, preferentially Kd or Ki as these represent true thermodynamic constants.

### 4.3 Implications for Model Evaluation

The substantial drop in performance observed with scaffold-based cross-validation (R^2^ = 0.52 to 0.41) suggests that standard random splitting leads to optimistic performance estimates. In random splits, structurally similar compounds (e.g., from the same lead optimization series) may appear in both training and test sets, allowing models to "memorize" chemical series rather than learn generalizable structure-activity relationships. This phenomenon has been well documented in the literature (Sheridan, 2013).

**Recommendation 2**: Use scaffold-based or temporal splits for model evaluation to obtain realistic estimates of predictive performance on novel chemical matter.

### 4.4 Recommendations for ML Practitioners

Based on our findings, we propose the following best practices for training binding affinity prediction models:

1. **Filter by measurement type**: Use only Kd or Ki values when possible; avoid mixing IC50 with equilibrium constants
2. **Cross-validate extreme values**: Flag and manually verify any training data points with |pKd| > 10 or |pKd| < 3
3. **Use scaffold-based splits**: Report scaffold-split performance as primary metrics; random-split metrics may be misleading
4. **Consider multi-database validation**: Compounds with consistent values across databases may be more reliable
5. **Report measurement type distribution**: Document the proportion of Ki/Kd/IC50 values in training data

### 4.5 Limitations

This study has several limitations:

1. **Ligand matching**: Our SMILES-based matching may miss legitimate matches due to stereochemistry or tautomer differences
2. **Protein identity**: We assumed PDB IDs represent identical proteins, but mutations or truncations may exist
3. **Temporal effects**: Database versions change over time; some conflicts may have been corrected in newer releases
4. **Single threshold**: Our 10-fold conflict threshold is somewhat arbitrary; sensitivity analysis with other thresholds would be informative

---

## 5. Conclusions

We performed the first systematic cross-database validation of binding affinity values between major structural biology databases. Our key findings are:

1. **25.7% of type-matched comparisons show >10-fold conflicts** between PDBbind and BindingDB
2. **Measurement type matching is critical**: Type-mismatched comparisons show 42-57% conflict rates
3. **Random CV overestimates performance**: Scaffold-based CV shows 21% lower R^2^ than random splits

These findings highlight the need for improved data curation practices and standardized reporting in the field. We recommend that researchers training binding affinity prediction models (1) filter data by measurement type, (2) validate extreme values against multiple databases, and (3) use scaffold-based splits for realistic performance evaluation.

Pre-trained models and analysis code are available at: [Repository URL to be added]

---

## Acknowledgments

[To be added]

---

## Author Contributions

[To be added]

---

## Conflicts of Interest

The authors declare no competing interests.

---

## Data Availability

All data and code used in this analysis are available at [GitHub repository URL]. Pre-trained models are provided as Python pickle files compatible with scikit-learn.

---

## References

Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *Journal of Medicinal Chemistry*, 39(15), 2887-2893. https://doi.org/10.1021/jm9602928

Brown, S. P., Muchmore, S. W., & Hajduk, P. J. (2009). Healthy skepticism: assessing realistic model performance. *Drug Discovery Today*, 14(7-8), 420-427. https://doi.org/10.1016/j.drudis.2009.01.012

Cheng, Y., & Prusoff, W. H. (1973). Relationship between the inhibition constant (K1) and the concentration of inhibitor which causes 50 per cent inhibition (I50) of an enzymatic reaction. *Biochemical Pharmacology*, 22(23), 3099-3108. https://doi.org/10.1016/0006-2952(73)90196-2

Gaulton, A., Hersey, A., Nowotka, M., Bento, A. P., Chambers, J., Mendez, D., ... & Leach, A. R. (2017). The ChEMBL database in 2017. *Nucleic Acids Research*, 45(D1), D945-D954. https://doi.org/10.1093/nar/gkw1074

Gilson, M. K., Liu, T., Baitaluk, M., Nicola, G., Hwang, L., & Chong, J. (2016). BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology. *Nucleic Acids Research*, 44(D1), D1045-D1053. https://doi.org/10.1093/nar/gkv1072

Jimenez-Luna, J., Grisoni, F., & Schneider, G. (2020). Drug discovery with explainable artificial intelligence. *Nature Machine Intelligence*, 2(10), 573-584. https://doi.org/10.1038/s42256-020-00236-4

Kalliokoski, T., Kramer, C., Vulpetti, A., & Gedeck, P. (2013). Comparability of mixed IC50 data - a statistical analysis. *PLoS One*, 8(4), e61007. https://doi.org/10.1371/journal.pone.0061007

Kramer, C., Kalliokoski, T., Gedeck, P., & Vulpetti, A. (2012). The experimental uncertainty of heterogeneous public Ki data. *Journal of Medicinal Chemistry*, 55(11), 5165-5173. https://doi.org/10.1021/jm300131x

Landrum, G. (2023). RDKit: Open-source cheminformatics. https://www.rdkit.org

Liu, T., Lin, Y., Wen, X., Jorissen, R. N., & Gilson, M. K. (2007). BindingDB: a web-accessible database of experimentally determined protein-ligand binding affinities. *Nucleic Acids Research*, 35(Database issue), D198-D201. https://doi.org/10.1093/nar/gkl999

Mendez, D., Gaulton, A., Bento, A. P., Chambers, J., De Veij, M., Felix, E., ... & Leach, A. R. (2019). ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Research*, 47(D1), D930-D940. https://doi.org/10.1093/nar/gky1075

Rifaioglu, A. S., Atas, H., Martin, M. J., Cetin-Atalay, R., Atalay, V., & Dogan, T. (2019). Recent applications of deep learning and machine intelligence on in silico drug discovery: methods, tools and databases. *Briefings in Bioinformatics*, 20(5), 1878-1912. https://doi.org/10.1093/bib/bby061

Sheridan, R. P. (2013). Time-split cross-validation as a method for estimating the goodness of prospective prediction. *Journal of Chemical Information and Modeling*, 53(4), 783-790. https://doi.org/10.1021/ci400084k

Wang, R., Fang, X., Lu, Y., & Wang, S. (2004). The PDBbind database: collection of binding affinities for protein-ligand complexes with known three-dimensional structures. *Journal of Medicinal Chemistry*, 47(12), 2977-2980. https://doi.org/10.1021/jm030580l

Yang, X., Wang, Y., Byrne, R., Schneider, G., & Yang, S. (2019). Concepts of artificial intelligence for computer-assisted drug discovery. *Chemical Reviews*, 119(18), 10520-10594. https://doi.org/10.1021/acs.chemrev.8b00728

---

## Supplementary Figures

### Figure S1. Molecular Property Distributions
![Figure S1](figures/Figure_S1_property_distributions.png)

**Figure S1.** Distribution of molecular properties (molecular weight, LogP, TPSA, hydrogen bond donors) across the PDBbind dataset. The dataset shows typical drug-like property distributions.

### Figure S2. Drug-Likeness Analysis
![Figure S2](figures/Figure_S2_drug_likeness.png)

**Figure S2.** Lipinski's Rule of Five analysis (Lipinski et al., 2001). Approximately 85% of PDBbind ligands satisfy all four Lipinski criteria, consistent with the focus on drug-like molecules.

### Figure S3. Affinity vs Molecular Properties
![Figure S3](figures/Figure_S3_affinity_properties.png)

**Figure S3.** Relationship between binding affinity and molecular properties. Weak correlations are observed, suggesting that simple property-based rules cannot predict binding affinity.

### Figure S4. Chemical Space Visualization
![Figure S4](figures/Figure_S4_chemical_space.png)

**Figure S4.** t-SNE visualization of chemical space covered by PDBbind ligands based on Morgan fingerprints (Rogers & Hahn, 2010). Distinct clusters correspond to different chemical scaffolds.

### Figure S5. Ligand Size Analysis
![Figure S5](figures/Figure_S5_size_analysis.png)

**Figure S5.** Distribution of ligand sizes (heavy atom count) in PDBbind. The majority of ligands contain 20-40 heavy atoms.

### Figure S6. Summary Dashboard
![Figure S6](figures/Figure_S6_summary_dashboard.png)

**Figure S6.** Complete analysis summary showing key metrics from the cross-database comparison and ML evaluation.

---

*Manuscript prepared for submission to [Journal Name]*
