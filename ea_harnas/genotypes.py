from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat channels')

EANAS =Genotype(normal=[('max_pool_3x1', 1), ('conv_3x3', 0), ('conv_7x7', 2), ('dil_conv_5x5', 1), ('max_pool_3x1', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], normal_concat=[5], channels=[48])
# uci
UCIBESTF1 = Genotype(normal=[('conv_1x1', 0), ('conv_3x1', 1), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x1', 3), ('max_pool_3x1', 3), ('skip_connect', 3), ('max_pool_3x1', 4)], normal_concat=[5], channels=[48])
UCIBEST = Genotype(normal=[('conv_1x1', 0), ('conv_3x1', 0), ('max_pool_3x1', 2), ('conv_3x1', 2), ('max_pool_5x1', 3), ('max_pool_3x1', 3), ('skip_connect', 3), ('max_pool_3x1', 4)], normal_concat=[5], channels=[40])
# uni
UNIBESTF1 = Genotype(normal=[('conv_9x1', 0), ('conv_3x1', 0), ('conv_3x1', 2), ('conv_1x1', 2), ('max_pool_5x1', 3), ('conv_5x1', 1), ('max_pool_5x1', 3), ('max_pool_3x1', 4)], normal_concat=[5], channels=[56])
UNIBEST = Genotype(normal=[('conv_7x1', 0), ('skip_connect', 0), ('max_pool_3x1', 2), ('conv_1x1', 0), ('max_pool_5x1', 3), ('conv_1x1', 0), ('max_pool_3x1', 3), ('max_pool_3x1', 4)], normal_concat=[5], channels=[56])
# kar
KARBESTF1 = Genotype(normal=[('skip_connect', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('conv_3x1', 2), ('max_pool_5x1', 1), ('max_pool_5x1', 0), ('max_pool_5x1', 3), ('max_pool_3x1', 3)], normal_concat=[4, 5], channels=[40])
KARBEST = Genotype(normal=[('conv_5x1', 1), ('max_pool_5x1', 1), ('max_pool_3x1', 2), ('max_pool_5x1', 2), ('max_pool_5x1', 1), ('conv_3x1', 2), ('max_pool_5x1', 3), ('conv_1x1', 4)], normal_concat=[5], channels=[40])
# wis
WISBESTF1 = Genotype(normal=[('skip_connect', 1), ('skip_connect', 1), ('max_pool_3x1', 2), ('skip_connect', 1), ('max_pool_5x1', 1), ('conv_9x1', 3), ('skip_connect', 2), ('max_pool_3x1', 4)], normal_concat=[5], channels=[40])
WISBEST = Genotype(normal=[('conv_1x1', 1), ('skip_connect', 1), ('max_pool_3x1', 2), ('max_pool_3x1', 0), ('max_pool_5x1', 0), ('skip_connect', 3), ('skip_connect', 4), ('max_pool_3x1', 4)], normal_concat=[5], channels=[40])
# opp
OPPBESTF1 = Genotype(normal=[('skip_connect', 1), ('max_pool_5x1', 1), ('conv_1x1', 0), ('conv_1x1', 2), ('max_pool_5x1', 0), ('max_pool_5x1', 1), ('max_pool_3x1', 1), ('max_pool_3x1', 4)], normal_concat=[3, 5], channels=[40])
OPPBEST = Genotype(normal=[('skip_connect', 1), ('max_pool_5x1', 1), ('max_pool_3x1', 2), ('skip_connect', 0), ('skip_connect', 1), ('max_pool_5x1', 2), ('max_pool_5x1', 1), ('max_pool_3x1', 4)], normal_concat=[3, 5], channels=[40])
# OPPBESTF1 = Genotype(normal=[('conv_7x1', 0), ('max_pool_3x1', 1), ('conv_1x1', 0), ('conv_1x1', 0), ('max_pool_5x1', 0), ('max_pool_5x1', 1), ('max_pool_5x1', 1), ('max_pool_3x1', 4)], normal_concat=[2, 3, 5], channels=[56])