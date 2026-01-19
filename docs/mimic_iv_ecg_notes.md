# MIMIC-IV-ECG Usage Notes

## Important Limitations

Based on [PhysioNet Usage Notes](https://physionet.org/content/mimic-iv-ecg/):

### Timestamp Synchronization Issues

1. **Internal Clock**: ECG date and time were recorded by the machine's internal clock, often **not synchronized** with external time sources.

2. **Database Mismatch**: This can lead to ECG timestamps being **out of sync** with other MIMIC-IV databases:
   - MIMIC-IV Clinical Database
   - MIMIC-IV Waveform Database

3. **Location-Based Issues**: Some ECGs were collected **outside the ED and ICU**, meaning their timestamps will **not overlap** with data from the MIMIC-IV Clinical Database.

### Implications for Research

- **Be cautious** when linking ECG records to clinical events based on timestamps alone
- **Verify** timestamp matches before making clinical correlations
- Consider that ECGs may have been taken before/during/after hospital admission
- Some ECGs may not have corresponding clinical data due to location/timing

## Timestamp Extraction

ECG records use WFDB format with:
- `base_date`: Date in WFDB format
- `base_time`: Time in WFDB format

These correspond to the timestamps for diagnostic ECGs provided in the summary tables.

### Example: Extracting Timestamps

```python
from src.data.ecg import extract_timestamp_from_record

# Single record
timestamp_info = extract_timestamp_from_record(
    "data/raw/demo/ecg/mimic-iv-ecg-demo/files/p10000032/s107143276/107143276"
)
# Returns: {'study': '107143276', 'date': '2113-08-25', 'time': '13:58:00'}
```

```bash
# Extract all timestamps from directory
python scripts/extract_ecg_timestamps.py \
  --data_dir data/raw/demo/ecg/mimic-iv-ecg-demo \
  --output outputs/timestamps.csv
```

## Linking to MIMIC-IV Clinical Database

To link ECG records to clinical data, you need:
1. `subject_id` (extracted from record path, e.g., `p10023771` → `10023771`)
2. `base_date` and `base_time` from the ECG record
3. Query Clinical Database for admissions matching `subject_id`

**Example SQL Query:**
```sql
SELECT * 
FROM `physionet-data.mimiciv_hosp.admissions` 
WHERE subject_id = 10023771
```

**Note**: Due to timestamp synchronization issues, you may need to:
- Use a time window around the ECG timestamp
- Check if ECG occurred before/during/after admission
- Verify matches manually for critical analyses

## Demo Dataset vs Full Dataset

This codebase is configured for the **MIMIC-IV-ECG Demo Dataset**, which contains:
- A subset of records for testing and development
- Same format and structure as the full dataset
- Same timestamp limitations apply

To use the full dataset, update paths in `configs/data/default_paths.yaml`.

