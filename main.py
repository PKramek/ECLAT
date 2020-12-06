import matplotlib.pyplot as plt
import logging
from Eclat.eclat import Eclat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

eclat = Eclat('data/data2.txt', min_support=0.0002, min_confidence=0.8)

eclat.eclat()
results_dataframe = eclat.get_results_dataframe()
eclat.save_results_to_csv('results/results.csv')
eclat.save_results_to_json('results/results.json')
eclat.save_results_to_excel('results/results.xlsx')
eclat.print_results_to_console()

print(results_dataframe.head())

plt.scatter(results_dataframe.lift, results_dataframe.cosine, c='r', s=5, alpha=0.5)
plt.grid()
plt.show()

plt.scatter(results_dataframe.lift, results_dataframe.certainty_factor, c='b', s=5, alpha=0.5)
plt.grid()
plt.show()

plt.scatter(results_dataframe.lift, results_dataframe.conviction, c='g', s=5, alpha=0.5)
plt.grid()
plt.show()
