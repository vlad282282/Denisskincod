import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def fun(a):
    if a=="none":
        return 0
    else:
        return 1

def make_result(x_train, x_test):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors = 75)
    classifier.fit(x_train, y_train)

    return classifier.predict(x_test)


df = pd.read_csv("train_df.csv")
#############################################################################################
#################################ДЕЛАЕМ СВОИ МАНИПУЛЯЦИИ######################################


#удаяляем лишние столбцы
df = df.drop(["gender","race/ethnicity",
        "parental level of education","lunch"],axis=1)

df["test preparation course"] = df["test preparation course"].apply(fun)


###################################################################################################
x = df.drop('result', axis = 1) # данные
y = df['result'] # результаты


x_train = x  # тренировачные данные
y_train = y  # тренировочный результат
x_test = pd.read_csv("test_df.csv") # Тестовые данные

##########################################################ДУБЛЯЖ

#удаляем лишние столбцы из тестового ДФ
x_test = x_test.drop(["gender","race/ethnicity",
        "parental level of education","lunch",],axis=1)#

x_test["test preparation course"] = x_test["test preparation course"].apply(fun)
##############################################################


result = pd.DataFrame({"result": make_result(x_train, x_test)})
print(result)

result.to_csv("result.csv",index = False)#сохраняем результат