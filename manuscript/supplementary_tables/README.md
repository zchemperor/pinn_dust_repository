# Supplementary Tables

Companion to "Wind-informed mathematical regularization for sparse Last Glacial Maximum dust-deposition reconstruction".
These files are supplied with the submission; no external repository is needed to read them.
Encoding: UTF-8 (BOM); CSV with headers. Empty numeric fields mean missing, never zero.
Source-candidate metadata must not be treated as verified attributes of current input records.

## File guide

- Table_S1_record_status.csv (317 rows): All 317 retained inputs, source-candidate counts and unresolved lineage status.
- Table_S2_source_candidates.csv (396 rows): All input/source-row candidate links. Archive, reference, age and size fields describe source candidates, never verified current identities.
- Table_S3_value_ratios.csv (118 rows): 118 exact-coordinate single-core/value candidate pairs; one lacks an LGM value. Ratios recomputed from unrounded numeric columns.
- Table_S4_historical_age_rows.csv (102 rows): 102 historical marine source rows; ages are not assigned to current inputs.
- Table_S5_historical_age_combinations.csv (29 rows): 29 interval combinations among 94 historical rows with both age bounds.
- Table_S6_input_row_mapping.csv (317 rows): Input-to-raw-row mapping, unit conversion and unresolved original metadata.
- Table_S7_nearby_memberships.csv (317 rows): All 317 rows and exploratory 3 km connected-component membership.
- Table_S8_nearby_groups.csv (299 rows): Connected components including singletons; group diameters may exceed the edge threshold.
- Table_S9_split_group_counts.csv (6 rows): Test counts and cross-partition proximity checks for the six splits.
- Table_S10_subset_memberships.csv (986 rows): Held-out membership by exact_flux93 / tolerance_flux98 candidate rule and complement.
- Table_S11_subset_counts.csv (24 rows): Uneven candidate coverage in each held-out split.
- Table_S12_subset_metrics.csv (312 rows): Descriptive candidate/complement errors on unstandardized log10 deposition; no intervals.
- Table_S13_subset_wind_contrasts.csv (264 rows): Wind-PINN minus comparator errors in the candidate/complement subsets; positive favors comparator.
- Table_S14_source_inventory.csv (6 rows): Historical source-table row and numeric-LGM counts.
- Table_S15_PaleoDust_candidates.csv (48 rows): Separate PaleoDust coordinate-only comparison; excluded from the 93/98 coverage counts.
- Table_S16_nearby_paired_contrasts.csv (1728 rows): Paired nearby-group error contrasts and conditional nominal intervals on fixed fits.

## Identifiers, units and interpretation

- record_id is the stable zero-based retained input index (0..316), preserving repeated input rows. raw_data_row_1based counts raw data rows excluding the header. source_data_row_1based is local to each source table, not a current-record identity.
- Coordinates (lon/lat, longitude/latitude and their candidate/input variants) are degrees. delta_lon_deg uses longitude wrapping. converted_g_m2_yr, input_g_m2_yr, candidate_LGM and candidate_HOL are fluxes in g m^-2 yr^-1 when numeric; raw_MAR/input_raw_MAR retain the upstream kg m^-2 s^-1 convention. Original measurement units remain unverified. dep is converted deposition; log_dep is its unstandardized log10 value relative to 1 g m^-2 yr^-1.
- candidate_* / core / archive / reference / age fields in Tables S2-S5 belong to historical source entries. Historical ages are ka; they are not transferred to present inputs. A source age_model is documentary text, not a validated chronology for an input row.
- Table S1 columns ending candidate_rows or coordinate_rows count candidate pairs satisfying that named rule. Coverage instead counts distinct record_id with a count greater than zero. combined means the union of historical DIRTMAP and Lambert Table S1, not PaleoDust. *_distinct_candidate_names count source names; multiplicities count input occurrences, not independent measurements.
- exact_coordinate means equality without rounding. Exploratory rectangular windows are <=0.05 degrees per coordinate for historical DIRTMAP and <=0.1 for Lambert Table S1. within_0p1percent means abs(input/source_LGM-1)<=0.001; within_0p2percent uses 0.002. Missing source LGM never meets flux agreement. relative_LGM_difference and input_over_candidate_LGM retain unrounded ratios. Candidate PM10 fractions/products are diagnostics only.
- Table S3 uses a DIFFERENT population: one distinct core-identifier/LGM-value candidate at an exact coordinate in the historical DIRTMAP tables, counting repeated appearances once, WITHOUT requiring flux agreement. Its input_over_source column is recomputed from the two unrounded flux columns, not the rounded diagnostic exported earlier. One of 118 rows lacks an LGM value. Values agreeing near one do not prove inherited identity; smaller loess ratios do not identify a particle-size threshold.
- identity_status remains UNRESOLVED_NO_VERIFIED_ROW_LINEAGE. numerical_evidence is a numerical relation, not identity confirmation. A unique coordinate does not establish measurement independence.
- component_3km is a connected-component label under an exploratory great-circle edge threshold of 3 km (Earth radius 6371 km); n, n_records and n_coordinates are counts, record_ids lists zero-based inputs, max_pair_distance_km is component diameter. This grouping is not an independence scale. Test group counts and cross_split fields refer to each named split.
- split names map to the main manuscript: outer_north_america / outer_east_asia / outer_tropical_atlantic_africa are the three regional tests; stress_leave_south / stress_leave_tropics / stress_leave_north_excluding_p2dev are the three extensive holdouts, with the northern development enclave excluded. The rule itself, rather than a name, defines membership in Table 1 of the main manuscript.
- exact_flux93 means exact coordinate plus <=0.1% flux agreement with either historical source; tolerance_flux98 uses the coordinate windows above plus the same flux rule. selection distinguishes candidate rows from their complement. The subsets overlap across rules, retain the original fits, and are partly selected by observed target values. They are not a clean or homogeneous validation set.
- estimand=ensemble evaluates the fixed five-seed mean prediction; mean_seed_metric averages five separate metrics. Deterministic predictors have one fitted prediction under either label. rmse, mae and bias are unstandardized log10-scale errors; r2 is dimensionless. bias is predicted minus observed. delta_rmse/delta_mae in Table S13 are Wind-PINN minus the named comparator. Table S16 uses A minus B: A_lower_error/A_higher_error/unresolved label the conditional interval, not a universal ranking. ci95_low/high are nominal paired percentile endpoints, conditional on fixed fits, without seed resampling or multiplicity adjustment. nearby_cluster retains record weights; nearby_equal assigns equal total weight to components. record_iid resamples individual rows; coordinate_cluster resamples exact-coordinate groups while retaining record weights.
- No candidate archive/age/size field was used to retrain a model or assign current identities. Historical version changes, original units, ages, particle sizes, measurement errors and co-located independence remain unresolved.

## File integrity

manifest.json lists each table's SHA-256, source SHA-256, original project-relative source path, row count and exact column names. Paths identify reproducibility inputs; they are not web URLs. No claim is made that these paths are already available in a public repository. Original publisher PDFs/Word supplements and database files are not bundled here.

## Coefficient and residual diagnostics (S17-S24)

independent evaluation on reconstructed point sets; not a recovery of the original training losses

- Table_S17_coefficients.csv (78 rows): Exact final scalar coefficients and raw parameters for 78 fixed fits.
- Table_S18_residual_diagnostics.csv (156 rows): 156 independent evaluations with signed decomposition and weighted loss fractions.
- Table_S19_point_set_inventory.csv (6 rows): Reconstructed and independent-grid counts and saved-array hashes.
- Table_S20_point_set_sensitivity.csv (78 rows): Paired grid/reconstructed ratios for each fixed fit.
- Table_S21_summary_by_split.csv (36 rows): Split-level descriptive residual and coefficient summaries.
- Table_S22_historical_loss_components.csv (11778 rows): Already-weighted historical losses; 78 fits times 151 saved times.
- Table_S23_historical_fraction_summary.csv (604 rows): Descriptive historical weighted PDE fraction summaries.
- Table_S24_numerical_checks.csv (156 rows): Prediction, decomposition and derivative implementation checks.

Tables S17-S21 and S24 describe independent evaluation; S22-S23 retain historical log origin. evaluation_notice preserves this distinction. run identifies a saved fit, family distinguishes primary and weight experiments, seed is a fit identifier, and pde_weight is the residual weight. Each primary family contains 30 fits across six splits; each additional weight contains nine across three extensive splits. These are individual-fit diagnostics, not ensemble metrics.

d_nd is the dimensionless scalar coefficient, raw_parameter its sigmoid input, d_nd_minus_lower its distance from 0.02, and lower_fraction=(d_nd-0.02)/1.98. near_lower uses <=1e-4. point_set is reconstructed or regular_grid. a_squared_mean=<A^2>, q_squared_mean=<q^2> at unit coefficient, a_q_product_mean=<Aq> is a signed product mean (not covariance), diffusion_squared_mean=D^2<q^2>, signed_cross_term=2D<Aq>, residual_squared_mean=<R^2>. conditional_gradient/curvature hold the fitted field fixed; an interior conditional optimum does not imply joint identifiability or a stationary final fit. Positive/negative values and full precision are retained.

weighted_residual=pde_weight*residual_squared_mean. weighted_nonpde sums weighted observation, seam and boundary terms; independent_composite_objective is their sum. pde_fraction is a fraction (multiply by 100 for percent), not a gradient fraction. cancellation_fraction=-signed_cross_term/(a_squared_mean+diffusion_squared_mean), preserving sign. Other *_mse terms are unweighted component means. *_ratio fields pair regular grid with reconstructed points for the same fit. Neither point set uses spherical area weighting. Grid evaluation changes only PDE points; non-PDE evaluations remain fixed. See Supplementary Methods S6 for exact definitions. Tables S22-S23 contain already-weighted historical values; their origin must not be conflated with independent evaluation.
