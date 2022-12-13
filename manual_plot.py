import matplotlib.pyplot as plt
import numpy as np


plot_56_gs = [0,0.25,0.5]
plot_56_fid = [1,0.998924,0.973925]

plot_136_gs = [1,1.5]
plot_136_fid = [0.937059,0.843385]

plot_176_gs = [2,2.5,3]
plot_176_fid = [0.820253,0.825067,0.816293]

plt.plot(plot_56_gs,plot_56_fid,"o",label = '56'+ " "+ r'$\mathbb{CS}_{K}$'+" states")
plt.plot(plot_136_gs,plot_136_fid,"+",label='136'+" "+ r'$\mathbb{CS}_{K}$'+" states")
plt.plot(plot_176_gs,plot_176_fid,"x",label='176'+" "+ r'$\mathbb{CS}_{K}$'+" states")
plt.xlabel(r'$g$')
plt.ylabel('Fidelity')
plt.legend()
plt.savefig('reverse_engineer_ansatz_results/diffansatzsize.pdf')
plt.close()