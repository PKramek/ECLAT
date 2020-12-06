import matplotlib.pyplot as plt

from Eclat.eclat import Eclat

eclat = Eclat('data/BMS1_itemset_mining.txt', min_support=0.001, min_confidence=0.9)

eclat.eclat()
results_dataframe = eclat.get_results_dataframe()
# eclat.save_results_to_csv('results/results.csv')
# eclat.save_results_to_json('results/results.json')
# eclat.save_results_to_excel('results/results.xlsx')

print(results_dataframe.head())

plt.scatter(results_dataframe.confidence, results_dataframe.cosine, c='r', s=5, alpha=0.5)
plt.grid()
plt.show()

plt.scatter(results_dataframe.confidence, results_dataframe.certainty_factor, c='b', s=5, alpha=0.5)
plt.grid()
plt.show()

plt.scatter(results_dataframe.confidence, results_dataframe.conviction, c='g', s=5, alpha=0.5)
plt.grid()
plt.show()
