import pstats
import os
from pstats import SortKey


filename = 'Data/Resultados/MonteCarloFEI/profile_stats'
output_log_path = os.path.join(os.path.dirname(filename), 'profile_stats.log')
with open(output_log_path, 'w') as f:
    p = pstats.Stats(filename, stream=f)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)
