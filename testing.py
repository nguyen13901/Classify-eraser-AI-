from sklearn.preprocessing import LabelBinarizer
import pickle
lb = LabelBinarizer()

a = [1, 2, 3, 1, 3, 4, 2, 1, 2]

a = lb.fit_transform(a)

validY = pickle.load(open('valid_labels.pkl', 'rb'))

validY = lb.fit_transform(validY)
print(validY)