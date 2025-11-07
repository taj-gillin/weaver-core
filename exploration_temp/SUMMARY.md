# Data Exploration Summary

## Overview
This exploration analyzed the Aleph dataset to understand the data structure, compute standardization parameters, and identify potential improvements for the classifier.

## Dataset Information
- **Sample file**: `/HEP/data/share/aleph/aleph-data/ntuples/mc/output_0_train.root`
- **Tree name**: `tree`
- **Total entries in sample**: 28,852 (analyzed first 10,000)
- **Total branches available**: 71
- **Variables currently in use**: 36
- **Unused variables**: 39

## Current Input Variables

### Point Cloud Variables (pf_points)
- `pfcand_px`, `pfcand_py`, `pfcand_pz`, `pfcand_e`

### Feature Variables (pf_features)
- Energy/momentum: `pfcand_pt_log`, `pfcand_e_log`, `pfcand_ptrel_log`, `pfcand_erel_log`
- Angular: `pfcand_drrel`, `pfcand_thetarel`, `pfcand_phirel`
- Charge: `pfcand_charge`
- Impact parameters: `pfcand_dxy`, `pfcand_dz`
- B-tagging: `pfcand_btagSip2dVal`, `pfcand_btagSip2dSig`, `pfcand_btagSip3dVal`, `pfcand_btagSip3dSig`, `pfcand_btagJetDistVal`, `pfcand_btagJetDistSig`
- Particle type flags: `pfcand_isChargedHad`, `pfcand_isNeutralHad`, `pfcand_isGamma`, `pfcand_isEl`, `pfcand_isMu`

## Key Findings

### Standardization Parameters
All input variables have been analyzed and standardization parameters (center and scale) have been computed using the median and robust scale (based on 16th and 84th percentiles).

**Notable observations:**
1. **B-tagging variables** (`pfcand_btag*`) have median values of -9.0, suggesting many particles have default/missing values. These need special handling.
2. **Impact parameters** (`pfcand_dxy`, `pfcand_dz`) also have median -9.0, indicating many particles without track information.
3. **Log-transformed variables** have reasonable distributions suitable for standardization.
4. **Particle type flags** are binary (0/1) and don't need standardization (center=0, scale=1).

### Unused Variables with Potential

Several unused variables were identified that might have discriminating power:

1. **Strange quark tagging variables** (for future use):
   - `recojet_isS` - Indicates if jet is strange quark
   - `recojet_isD` - Indicates if jet is down quark
   - `recojet_isU` - Indicates if jet is up quark
   - Note: These could be used for future strange tagging work

2. **Particle-level variables**:
   - `pfcand_Bz` - Bz value (could be related to magnetic field)
   - `pfcand_erel`, `pfcand_ptrel` - Non-log versions of relative energy/momentum
   - Various covariance matrix elements (`pfcand_detadeta`, `pfcand_dphidphi`, etc.)

3. **Jet-level counting variables**:
   - `nphotons` - Number of photons in jet
   - `nchargedhad` - Number of charged hadrons
   - `nneutralhad` - Number of neutral hadrons
   - `nel`, `nmu` - Number of electrons/muons

4. **Other variables**:
   - `bz` - Bz value at jet level
   - `recojet_flavour` - Jet flavour ID

## Generated Files

1. **`data_config_pnet_standardized.yaml`**: Updated config file with standardization parameters for all input variables
2. **`exploration_report.txt`**: Detailed statistics report for all variables

## Next Steps

### Immediate Actions (Data Preprocessing)
1. ✅ **Standardization parameters computed** - All variables now have center and scale values
2. **Review and test the standardized config** - Verify the updated config works correctly
3. **Update the data config** - Replace the current config with the standardized version (or create a new one)

### Future Improvements
1. **Variable selection**: Test if any unused variables improve classifier performance
2. **Strange tagging**: When ready, split `udsg` category into separate `uds` and `g` categories, potentially using `recojet_isS` and related variables
3. **Architecture improvements**: Consider ParticleNet or ParticleTransformer variations
4. **Additional preprocessing**: Consider if any other transformations would help (e.g., different normalizations for different variable types)

## Standardization Implementation

The standardization follows the weaver framework format:
```yaml
vars:
  - [var_name, center, scale, clip_min, clip_max, pad_value]
```

Where:
- **center**: Median value (subtracted from each value)
- **scale**: Robust scale factor (1 / (max(84th_percentile - median, median - 16th_percentile)))
- **clip_min/max**: Bounds for clipping outliers (default: -5, 5)
- **pad_value**: Value used for padding (default: 0)

## Notes

- The b-tagging and impact parameter variables with median -9.0 suggest a default/missing value convention. These should be handled carefully during standardization.
- The exploration used 10,000 events from the training sample. For production, consider using a larger sample or the full dataset to compute more robust statistics.
- The script is in `exploration_temp/` and can be removed after review.



