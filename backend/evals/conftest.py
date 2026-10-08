# The fixture repos under tasks/ contain tests that are *meant* to fail until a
# patch is applied, so pytest must not collect them directly.
collect_ignore = ["tasks"]
