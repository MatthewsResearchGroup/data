import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from os import makedirs
import seaborn as sns

makedirs ('plots', exist_ok=True)

def plot(df, basis, title, **kwargs):
    colors = [cm.gist_rainbow(x) for x in np.linspace(0.0,0.75,11)]
    kcal=627.509474064812252
    ax = (kcal*df.xs(basis,level='Basis',axis=1)).plot((plt.xlabel('IRC Point')), (plt.ylabel('Error (kcal/mol)')), color=colors)
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left", title="-log(ε)")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.hlines([-1,1], -10000, 10000, linestyles='dashed')
    ax.set_xlim([min(df.index),max(df.index)])
    ax.set_ylim([-7,7])
    plt.savefig(f'plots/{title.replace("/","_")}.pdf', dpi=300, bbox_inches='tight')
  #  rms = np.sqrt(np.square(data).mean(axis=0))
  #  rmd = data.max(axis=0) - data.min(axis=0)
  #  sd  = data.std(axis=0)
  # data2 = pd.DataFrame(columns=['rms', 'rmd', 'sd'], data=np.array([rms, rmd, sd]).T)
  # axis2 =(627*data2).plot.bar(legend=False, ylabel='Error (kcal/mol)', color=colors)
  # plt.savefig(f'plots/{title.replace("/","_")}_stats.pdf', dpi=300, bbox_inches='tight')
  # return data2.to_numpy()

for name in ['data.csv']:
    data = pd.read_csv(name)
    index = ''
    for tag in ['Angle', 'Step']:
        if tag in data.columns:
            index = tag
            break

    data = data.pivot_table(index=index, columns=['Method', 'X', 'Basis'], values='Energy')
    data = data-data.iloc[0]
    mp2a = data.xs('mp2a',level='Method',axis=1).sub(data.xs(('mp2',1.0),level=('Method','X'),axis=1),axis=0)
    mp2b = data.xs('mp2b',level='Method',axis=1).sub(data.xs(('mp2',1.0),level=('Method','X'),axis=1),axis=0)
    mp3b = data.xs('mp3b',level='Method',axis=1).sub(data.xs(('mp3',1.0),level=('Method','X'),axis=1),axis=0)
    mp3d = data.xs('mp3d',level='Method',axis=1).sub(data.xs(('mp3',1.0),level=('Method','X'),axis=1),axis=0)
    ccsd = data.xs('ccsdthc',level='Method',axis=1).sub(data.xs(('ccsd',1.0),level=('Method','X'),axis=1),axis=0)

    basis_name={'dz':'cc-pVDZ', 'adz':'aug-cc-pVDZ', 'tz':'cc-pVTZ'}
    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product([['dz', 'adz', 'tz'], \
                                                              ['rms', 'rmd', 'sd'], \
                                                              ['mp2a', 'mp2b', 'mp3b', 'mp3d', 'ccsd', 'ccsd-mp2b-mp3d', 'ccsd-mp2b', 'mp2b-mp2a', 'mp3d-mp3b']], \
                                                             names=['Basis', 'Type', 'Plot']), \
                          index=range(33))

    for basis in ['dz', 'adz', 'tz']:
        df_all.loc[:, (basis, slice(None), 'mp2a')] = plot(mp2a, basis, title=f'{name} mp2a/{basis_name[basis]}')
        df_all.loc[:, (basis, slice(None), 'mp2b')] = plot(mp2b, basis, title=f'{name} mp2b/{basis_name[basis]}')
        df_all.loc[:, (basis, slice(None), 'mp3b')] = plot(mp3b, basis, title=f'{name} mp3b/{basis_name[basis]}')
        df_all.loc[:, (basis, slice(None), 'mp3d')] = plot(mp3d, basis, title=f'{name} mp3d/{basis_name[basis]}')
        df_all.loc[:, (basis, slice(None), 'ccsd')] = plot(ccsd, basis, title=f'{name} ccsd/{basis_name[basis]}')
        df_all.loc[:, (basis, slice(None), 'ccsd-mp2b-mp3d')] = plot(ccsd-mp2b-mp3d, 'dz', title=f'{name} ccsd-mp2b-mp3d/{basis_name[basis]}')
        df_all.loc[:, (basis, slice(None), 'ccsd-mp2b')] = plot(ccsd-mp2b, 'dz', title=f'{name} ccsd-mp2b/{basis_name[basis]}')
        df_all.loc[:, (basis, slice(None), 'mp2b-mp2a')] = plot(mp2b-mp2a, 'dz', title=f'{name} mp2b-mp2a/{basis_name[basis]}')
        df_all.loc[:, (basis, slice(None), 'mp3d-mp3b')] = plot(mp3d-mp3b, 'dz', title=f'{name} mp3d-mp3b/{basis_name[basis]}')

    plot_data = df_all.loc[:, (slice(None), 'rms', 'ccsd')]
    svm = sns.heatmap(plot_data)
    figure = svm.get_figure()
    title = name
    figure.savefig(f'plots/{title.replace("/","_")}_heatmap.pdf', dpi=300, bbox_inches='tight')
#plt.show()
