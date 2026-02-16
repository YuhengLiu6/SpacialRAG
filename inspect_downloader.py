import habitat_sim.utils.datasets_download as dd

print("Available UIDs:")
if hasattr(dd, 'data_sources'):
    for key in dd.data_sources.keys():
        print(f":: {key}")
else:
    print("Could not find data_sources in module.")
    print(dir(dd))
