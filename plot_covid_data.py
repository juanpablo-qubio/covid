import os
from os.path import join, exists
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import shutil

class COVID(object):
    """
    COVID data representation class
    """
    
    def __init__(self):
        """
        Class initializer
        """
        

    def run(self):
    
        df = self.get_data()
        countries = self.ask_for_countries(df)
        dfs, m_values = self.create_dfs(df, countries)
        self.plot_data(dfs, m_values)
        
    
    def get_data(self):

        date = datetime.today().strftime('%Y-%m-%d')

        # Download data
        try:
            get_xls = "wget -q https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-%s.xl" % date
            day = int(date.split("-")[2])
            os.system(get_xls)

            xls_file = 'COVID-19-geographic-disbtribution-worldwide-%s.xls' % date
            xls_file = join(os.getcwd(), xls_file)
            
            if not exists(xls_file):

                pday = "-" + str(day-1)
                date = date.replace("-" + str(day), pday)
                
                get_xls = "wget -q https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-%s.xls" % date
                os.system(get_xls)
                
                xls_file = 'COVID-19-geographic-disbtribution-worldwide-%s.xls' % date
                if exists(xls_file):
                    data_file = join(os.getcwd(), 'data', 'COVID_data.xls')
                    shutil.move(xls_file, data_file)
                    df = pd.read_excel(open(data_file, 'rb'))
                    print("Got data for date:", date)

                    return df
            else:
                data_file = join(os.getcwd(), 'data', 'COVID_data.xls')
                shutil.move(xls_file, data_file)
                df = pd.read_excel(open(data_file, 'rb'))
                print("Got data for date:", date)

                return df
                
        except Exception as err:
            print("Error:", err)
            

    def ask_for_countries(self, df):

        # Arrange df for countries selections
        
        print("\nSelect Country/es:")
        countries = sorted(np.unique(df['CountryExp'].values))

        n_countries = len(countries)
        g_size = int(n_countries/20)

        i = 0
        sep = "\t\t\t\t"
        for country in enumerate(countries):
            if i < len(countries)-1:
                print(i+1, "-", countries[i][0:20], sep, i+2, "-", countries[i+1])
            else:
                print(i+1, "-", countries[i-2])
                break
            i += 2

        countries = ['Spain', 'Italy', 'France', 'Iran']

        return countries
        

        
    def set_black_axis(self, axis):
        """
        Configure axis as white over black
        """

        axis.set_facecolor('black')
        axis.spines['bottom'].set_color('white')
        axis.spines['top'].set_color('white')
        axis.spines['right'].set_color('white')
        axis.spines['left'].set_color('white')
        axis.yaxis.label.set_color('white')
        axis.xaxis.label.set_color('white')
        axis.tick_params(axis='x', colors='white')
        axis.tick_params(axis='y', colors='white')
        # axis.set_title(title, color='white', fontsize=12)

        return axis

    def set_black_legend(self, ax):
        
        legend = ax.legend(frameon = 1)
        frame = legend.get_frame()
        frame.set_facecolor('black')
        frame.set_edgecolor('white')
        for text in legend.get_texts():
            text.set_color("white")
    
    def create_dfs(self, df, countries):
        """
        Create data frames
        """
        
        dfs = []
        m_values = 0
        for countrie in countries:
            df_c = df.loc[df['CountryExp'] == countrie]
            df_c = df_c.sort_values(by=['NewConfCases'])
            df_c = df_c[df_c['NewConfCases'] > 0]
            dfs.append(df_c)

            n_values = len(df_c['NewConfCases'].values)
            if n_values > m_values:
                m_values = n_values
            
        return dfs, m_values

    def plot_data(self, dfs, m_values):

        l_pct = []
        val = 0
        for n in range(m_values):
            val += (val + n)*0.33
            l_pct.append(val)

        cols = 3
        rows = 1

        fig = plt.figure(1, figsize=(20, 5))

        max_cases = 0
        for i in range(cols):

            ax = fig.add_subplot(rows, cols, i+1)
            ax = self.set_black_axis(ax)
            ax.set_xlabel("Days since first case")

            if i == 0:
                ax.set_ylabel("New cases")
            else:
                ax.set_ylabel("Total cases")

            for j, df in enumerate(dfs):

                # print("DF:", df.head())

                y_values = df['NewConfCases'].values
                x_values = np.arange(len(y_values))
                country = df['CountryExp'].values[0]
                cases = np.max(np.cumsum(y_values))
                if cases > max_cases:
                    max_cases = cases
            
                if i == 0:
                    ax.plot(x_values, y_values, label=country) #, color="y")
                elif i == 1:
                    ax.plot(x_values, np.cumsum(y_values), label=country) #, color="y")
                else:
                    if j == 0:
                        ax.plot(range(m_values), np.cumsum(l_pct), '--', label="33% Increase", color="w")

                    ax.plot(x_values, np.cumsum(y_values), label=country) #, color="y")

                    
            legend = self.set_black_legend(ax)

            ax.grid()
            if i > 0:
                ax.set_ylim(1, max_cases)
            if i == 2:
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
                ax.yaxis.set_minor_formatter(NullFormatter())

        # plt.show()
        fig_name = join(os.getcwd(), 'COVID_graphs.png')
        fig.savefig(fig_name, bbox_inches='tight', facecolor='k', edgecolor='k')
        print("Created figure:", fig_name)           


if __name__ == "__main__":
    worker = COVID()
    worker.run()