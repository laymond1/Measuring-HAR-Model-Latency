from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')

# uci
# UCIBESTF1 = Genotype(normal=[('conv_7x1', 0), ('conv_7x1', 1), ('max_pool_5x1', 1), ('max_pool_5x1', 2), ('max_pool_5x1', 1), ('conv_3x1', 0), ('max_pool_5x1', 0), ('max_pool_5x1', 3)], normal_concat=range(2, 6))
# UCIBESTF1 = Genotype(normal=[('conv_9x1', 0), ('diconv_3x1', 1), ('diconv_5x1', 2), ('conv_7x1', 0), ('diconv_5x1', 2), ('conv_7x1', 3), ('avg_pool_5x1', 4), ('conv_7x1', 1)], normal_concat=range(2, 6))
# UCICPU = Genotype(normal=[('conv_7x1', 1), ('conv_5x1', 0), ('conv_7x1', 0), ('avg_pool_5x1', 2), ('avg_pool_5x1', 3), ('avg_pool_5x1', 2), ('avg_pool_5x1', 3), ('avg_pool_5x1', 4)], normal_concat=range(2, 6))
# UCIA31 = 
# UCIAS10 = 
# uni
# UNIBESTF1 = Genotype(normal=[('max_pool_3x1', 0), ('conv_7x1', 1), ('diconv_5x1', 1), ('max_pool_3x1', 0), ('max_pool_3x1', 0), ('max_pool_3x1', 3), ('diconv_5x1', 1), ('max_pool_3x1', 0)], normal_concat=range(2, 6))
# UNICPU = Genotype(normal=[('avg_pool_5x1', 0), ('conv_3x1', 1), ('max_pool_3x1', 0), ('avg_pool_5x1', 2), ('diconv_3x1', 1), ('conv_9x1', 2), ('diconv_5x1', 1), ('conv_9x1', 0)], normal_concat=range(2, 6))
# UNIA31 = 
# UNIS10 = 
# kar
# KARBESTF1 = Genotype(normal=[('diconv_3x1', 1), ('conv_1x1', 0), ('conv_7x1', 2), ('conv_7x1', 1), ('conv_5x1', 1), ('max_pool_5x1', 3), ('max_pool_5x1', 3), ('max_pool_5x1', 2)], normal_concat=range(2, 6))
# KARCPU = 
# KARA31 = 
# KARS10 = 
# wis
# WISBESTF1 = Genotype(normal=[('conv_3x1', 1), ('conv_1x1', 0), ('conv_3x1', 0), ('max_pool_5x1', 2), ('conv_3x1', 1), ('diconv_5x1', 3), ('max_pool_5x1', 2), ('max_pool_5x1', 3)], normal_concat=range(2, 6))
# WISCPU = Genotype(normal=[('diconv_5x1', 0), ('diconv_3x1', 1), ('max_pool_5x1', 1), ('conv_1x1', 2), ('conv_5x1', 3), ('max_pool_5x1', 2), ('conv_9x1', 4), ('conv_5x1', 3)], normal_concat=range(2, 6))
# WISA31 = 
# WISS10 = 
# opp
OPPDNAS = Genotype(normal=[('avg_pool_5x1', 0), ('avg_pool_5x1', 1), ('diconv_3x1', 1), ('conv_7x1', 2), ('diconv_3x1', 3), ('conv_1x1', 2), ('diconv_3x1', 0), ('conv_9x1', 2)], normal_concat=range(2, 6))
OPPCPU = Genotype(normal=[('conv_5x1', 1), ('max_pool_3x1', 0), ('diconv_3x1', 1), ('conv_1x1', 0), ('diconv_3x1', 2), ('conv_5x1', 3), ('conv_1x1', 0), ('conv_5x1', 2)], normal_concat=range(2, 6))
# OPPBESTF1 = 
OPPA31 = Genotype(normal=[('conv_1x1', 1), ('max_pool_3x1', 0), ('avg_pool_5x1', 2), ('avg_pool_5x1', 0), ('diconv_3x1', 1), ('diconv_5x1', 2), ('conv_1x1', 1), ('avg_pool_5x1', 4)], normal_concat=range(2, 6))
OPPS10 = Genotype(normal=[('max_pool_3x1', 0), ('conv_1x1', 1), ('avg_pool_5x1', 0), ('avg_pool_5x1', 2), ('conv_3x1', 2), ('diconv_3x1', 1), ('conv_1x1', 1), ('diconv_3x1', 0)], normal_concat=range(2, 6))
# OPPBESTF1 = Genotype(normal=[('conv_7x1', 0), ('max_pool_3x1', 1), ('conv_1x1', 0), ('conv_1x1', 0), ('max_pool_5x1', 0), ('max_pool_5x1', 1), ('max_pool_5x1', 1), ('max_pool_3x1', 4)], normal_concat=[2, 3, 5], channels=[56])