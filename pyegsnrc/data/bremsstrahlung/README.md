# Data format

The original header in the original `nrc_brems.data` file says the following:

```python
# BREMSSTRAHLUNG CROSS-SECTION, TOTAL: (B^2/Z^2)*k*dsig/dk (mb). F. Tessier 2007.
# FORMAT:
#     nT nk; Tvalues; kvalues; "# BREMX.DAT";
#     100 blocks Z=1..100, nk lines of nT values
```

Data values have been left untouched. Instead, each table of values for each
element `Z` has been placed into its own `CSV` file. The columns have been
labelled what was called `nT` in the previous file, and the rows to `nK`.
