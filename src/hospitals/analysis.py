import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text

FILES = ['general', 'prenatal', 'sports']
ORDINALS = ['1st', '2nd', '3rd', '4th', '5th']
DIR = '../../data/'


class Analysis:

    def main(self, data_dir='../../data/'):
        frames = self.load_data(data_dir)
        df = pd.concat(frames, ignore_index=True)
        df.drop(df.columns[0], axis=1, inplace=True)
        df = self.preprocess(df)
        answers = self.visualize(df)
        for i, answer in enumerate(answers):
            print(f'The answer to the {ORDINALS[i]} question: {answer}')

    @staticmethod
    def load_data(dir: str) -> list[pd.DataFrame]:
        frames = [pd.read_csv(f'{dir + file}.csv') for file in FILES]
        keys = frames[0].keys()
        for frame in frames[1:]:
            frame.columns = keys
        return frames

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        df.dropna(how='all', inplace=True)
        df.fillna({'gender': 'f'}, inplace=True)
        df.replace({'gender': {'female': 'f', 'male': 'm', 'woman': 'f', 'man': 'm'}}, inplace=True)
        for col in ['bmi', 'diagnosis', 'blood_test', 'ecg', 'ultrasound', 'mri', 'xray', 'children', 'months']:
            df.fillna({col: 0}, inplace=True)
        return df

    @staticmethod
    def evaluate(df: pd.DataFrame) -> list[str]:
        answers = [df.value_counts('hospital').idxmax(),
                   f'{df[df['hospital'] == 'general'].value_counts('diagnosis', normalize=True)['stomach']:0.3f}',
                   f'{df[df['hospital'] == 'sports'].value_counts("diagnosis", normalize=True)['dislocation']:0.3f}']
        age_medians = df.groupby('hospital')['age'].median()
        answers.append(f'{int(age_medians['general'] - age_medians['sports'])}')
        most_bloodtests = df[df['blood_test'] == 't'].value_counts('hospital')
        answers.append(f'{most_bloodtests.index[0]}, {most_bloodtests.iloc[0]} blood tests')
        return answers

    def visualize(self, df: pd.DataFrame) -> list[str]:
        hist = self.age_hist(df)
        age_counts, maxbin_low, maxbin_high = hist[1], hist[0].argmax(), hist[0].argmax() + 1
        answers = [f'{int(age_counts[maxbin_low])}-{int(age_counts[maxbin_high])}']
        pie_texts = self.diagnosis_pie(df)
        answers.append(pie_texts[0].get_text())  # wedges are ordered by value, so text 0 is for the largest
        answers.append(self.height_distributions(df))
        return answers

    @staticmethod
    def age_hist(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, any]:
        plt.figure(1)
        hist = plt.hist(df['age'], bins=[0, 15, 35, 55, 70, 80])
        plt.xlabel('Age')
        plt.ylabel('Number of patients')
        plt.title('Distribution of patients by age')
        plt.show()
        return hist

    @staticmethod
    def diagnosis_pie(df: pd.DataFrame) -> list[Text]:
        plt.figure(2)
        pie = plt.pie(df['diagnosis'].value_counts(), labels=df['diagnosis'].value_counts().index, autopct='%1.1f%%')
        plt.title('Distribution of diagnoses')
        plt.show()
        return pie[1]  # returns the texts of the wedges

    @staticmethod
    def height_distributions(df: pd.DataFrame) -> str:
        plt.figure(3)
        data_list = [df[df['hospital'] == hospital]['height'] for hospital in df['hospital'].unique()]
        plt.violinplot(data_list)
        plt.show()
        return 'Find the answer in the plot'


if __name__ == '__main__':
    Analysis().main()
