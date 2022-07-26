import streamlit as st 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()  # pour modifier le thème

import config

from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



df = pd.read_csv("bank.csv")

# Réalisation des modifications évoquées précédemment sur un nouveau dataframe nommé df_clean
df_clean = df

# On renomme les 'unknown' de job en 'other'
df_clean['job'].replace('unknown', 'other', inplace=True)

# On renomme les 'unknown' de education en 'other'
df_clean['education'].replace('unknown', 'other', inplace=True)

# On renomme les 'unknown' de contact en 'other type of contact'
df_clean['contact'].replace('unknown', 'other type of contact', inplace=True)

# On supprime les deux clients ayant participé à la campagne précédente mais n'ayant pas de résultat renseignée à poutcome
df_clean.drop(df_clean.loc[ (df_clean['previous'] > 0)
              & (df_clean['poutcome'] == 'unknown') ].index, inplace=True)
# 2 lignes supprimées

# On renomme les 'unknown' de poutcome en 'not contacted previously'
df_clean['poutcome'].replace('unknown', 'not contacted previously', inplace=True)

# On supprime les clients ayant des valeurs aberrantes dans campaign (campaign >= 10)
df_clean.drop(df_clean.loc[(df_clean['campaign'] >= 10)].index, inplace=True)
# 262 lignes supprimées

# Pour la dataviz, à supprimer pour la modélisation
df_clean["classe_age"]=pd.cut(df_clean.age,bins=[18,29,39,49,59,95],labels=['18-29 ans','30-39 ans','40-49 ans','50-59 ans','Plus de 60 ans'])
df_clean["classe_balance"]=pd.cut(df_clean.balance,bins=[-7000,0,800,2000,4000,90000],labels=['<=0',']0,800]',']800,2000]',']2000,4000]', ">4000"])
df_clean.astype({'classe_age': 'category', 'classe_balance': 'category'})



# STREAMLIT
st.sidebar.title("Sommaire")

pages = ["Le projet","Analyse & nettoyage du jeu de données","Quelques visualisations","Préparation du jeu de données","Modélisation","Conclusion"]
page = st.sidebar.radio("Aller vers", pages)

st.sidebar.markdown("---")
st.sidebar.title("Auteurs")
for member in config.TEAM_MEMBERS:
    st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)



if page==pages[0]:
    st.title("Py4Bank")
    
    ## image
    st.image('Py4Bank.jfif')
    st.caption("A data analysis dashboard by Taly ATTIAS, Pierre-Loïc DUPONT, Tiphaine MOYNE, Steve SERVAIS")

    st.write("## Le projet")
    st.markdown("Ce projet a été réalisé dans le cadre de notre formation de Data Analyst via l'organisme *Datascientest*. Dans ce cadre, nous analyserons un jeu de données marketing d’une banque. Celui-ci contient des données personnelles de clients qui ont été appelés lors d'une campagne marketing afin de souscrire à un “dépôt à terme”. Lors de la souscription à ce produit, le client place un certain montant sur un compte spécifique et ne pourra pas toucher ces fonds avant l’expiration du terme. En échange, le client reçoit des intérêts de la part de la banque à la fin du terme.")
    st.markdown("L’objectif du projet est de définir un algorithme de machine learning (en Python) qui nous permettra de prédire si oui ou non un client souscrira au produit.")
    st.markdown("Le jeu de données est disponible sur [Kaggle](https://www.kaggle.com/janiobachmann/bank-marketing-dataset)")
    #st.image("bank.jpg")



elif page==pages[1]:
    st.header("Analyse & nettoyage du jeu de données")
    st.subheader("Descriptif des variables")
    st.write("Le jeu de données contient 11 162 lignes (donc autant de clients) et comprend 17 variables dont la variable cible (*deposit*).")

    if st.checkbox("Afficher un aperçu du jeu de données"):
        st.dataframe(df)

    st.write("**Variables quantitatives :**")
    st.write("- *Age* : âge du client")
    st.write("- *Day* : jour du dernier contact de la campagne en cours") 
    st.write("- *Balance* : montant des encours détenus par le client")
    st.write("- *Duration* : durée (en secondes) du dernier contact de la campagne en cours")
    st.write("- *Campaign* : nombre de contacts effectués au cours de cette campagne (inclut le dernier contact)")
    st.write("- *Pdays* : nombre de jours qui se sont écoulés depuis que le client a été contacté pour la dernière fois lors d’une campagne précédente")   
    st.write("- *Previous* : nombre de contacts effectué avant cette campagne et pour ce client")

    st.write("**Variables catégorielles avec plus de 2 catégories :**")   
    st.write("- *Job* : type d’emploi (12 valeurs possibles)")
    st.write("- *Marital* : statut marital (3 valeurs possibles : divorced, maried, single)")   
    st.write("- *Education* : niveau d’études (4 valeurs possibles : primary, secondary, tertiary et unknown)")
    st.write("- *Contact* : canal de communication lors du dernier contact de la campagne en cours (3 valeurs possibles : cellular, telephone, unknown)") 
    st.write("- *Month* : mois du dernier contact de la campagne en cours (12 valeurs possibles : jan, feb, mar…)")
    st.write("- *Poutcome* : résultat de la campagne marketing précédente (4 valeurs possibles :failure, other, success, unknown)") 

    st.write("**Variables catégorielles binaires (valeurs possibles yes/no) :**")  
    st.write("- *Default* : indique si le client a eu un défaut de paiement sur un crédit")
    st.write("- *Housing* : détention d’un prêt immobilier")
    st.write("- *Loan* : détention d’un prêt personnel")
    st.write("- *Deposit* (variable cible) : souscription ou non du client à un dépôt à terme") 
   
    st.markdown("Une étude plus précise de chaque variable est disponible [ici](https://docs.google.com/spreadsheets/d/1zWWpq5-Fe3lOffkXkgDtVQhWyTn2kAHMTxMvjyBvVzw/edit?usp=sharing)")

    st.subheader("Nettoyage du jeu de données")
    st.write("**Gestion des NaNs :**")  
    st.write("Le jeu de données ne contient aucun NaNs. Cependant 4 variables contiennent des 'unknown' : job, education, contact et poutcome que nous avons donc décidé de renommer comme suit :")          
    st.write("- job : “other”")
    st.write("- education : “other”")
    st.write("- contact : “other type of contact”")
    st.write("- poutcome : “not contacted previously”")

    st.write("**Suppression des valeurs aberrantes (264 lignes)**")  
    st.write("Nous avons pu identifier 2 lignes concernant des clients ayant participé à la campagne précédente mais sans résultat renseigné à poutcome. Nous les avons donc supprimées.")
    st.write("Par ailleurs, nous avons remarqué que la variable campaign contenait des valeurs supérieures ou égales à 10. Le client aurait alors été contacté 10 fois ou plus lors de cette campagne. Nous avons considéré ces données comme des valeurs aberrantes et avons donc supprimé les lignes correspondantes (262 lignes).")



elif page==pages[2]:
    st.header("Quelques visualisations")

    st.write("Nous avons réalisé un certain nombre de graphiques pour mieux comprendre notre jeu de données. Nous avons ainsi analysé chacune des variables mais surtout le croisement entre notre variable cible « deposit » et les autres variables du jeu de données. Pour les variables qui nous semblaient corrélées au vu des graphiques, nous avons réalisé des tests statistiques pour confirmer l’éventuelle dépendance entre les variables.")
    st.write("La répartition de la variable cible est assez équilibrée avec un **taux de souscription à 48%.**")
    st.image("Tx souscription.jpg")
    
    st.write("Nous avons organisé notre Dataviz en 3 thématiques et un client cible :")
                 
    
    socio_demo=st.checkbox("Caractéristiques socio-démographiques des clients")
    
    if socio_demo :
        
        st.subheader("Caractéristiques socio-démographiques des clients")
     
    
        fig_age=plt.figure(figsize=(12,5))
        sns.countplot(x="classe_age",data=df_clean, edgecolor="black", alpha=0.7,hue="deposit")
        plt.title("Souscription de dépôt à terme par classe d'âge")
        plt.xlabel("Classe d'âge")
        plt.ylabel('Effectif')
        st.pyplot(fig_age)
        
    
        fig_job = plt.figure(figsize=[12,5])
        sns.countplot(x='job', hue='deposit',edgecolor="black", alpha=0.7, data=df_clean)
        plt.title("Souscription de dépôt à terme par profession")
        plt.xlabel("Job")
        plt.ylabel('Effectif')
        plt.xticks(rotation=30)
        st.pyplot(fig_job)
        
        st.write("Nous remarquons ainsi que les **18-29 ans et les plus de 60 ans semblent les plus appétents** au dépôt à terme avec respectivement des taux de souscription de 60% et 77%.Ce qui se confirme lorsque l’on regarde le taux de souscription par profession : en effet, **les étudiants et les retraités** semblent plus enclins que les autres  à souscrire à un dépôt à terme. Les moins intéressés étant les ouvriers et les entrepreneurs.")


        fig_etude = plt.figure(figsize=[12,5])
        sns.countplot(x='education', hue='deposit',edgecolor="black", alpha=0.7, data=df_clean,order=("other","primary","secondary","tertiary"))
        plt.title("Souscription de dépôt à terme par niveau d'études")
        plt.xlabel("Niveau d'études")
        plt.ylabel('Effectif')
        st.pyplot(fig_etude)
        
        st.write("Les clients ayant un niveau d’étude supérieur ont des taux de souscription plus élevé que les autres.")
                
        
        fig_marital = plt.figure(figsize=[12,5])
        sns.countplot(x='marital', hue='deposit',edgecolor="black", alpha=0.7, data=df_clean,order=("single","married","divorced"))
        plt.title("Souscription de dépôt à terme par statut marital")
        plt.xlabel("Statut marital")
        plt.ylabel('Effectif')
        st.pyplot(fig_marital)
        
        st.write("Les personnes célibataires souscrivent davantage à un dépôt à terme plutôt que les personnes mariées ou divorcées.")

    equipement_bancaire=st.checkbox("Équipement bancaire des clients")
        
    if equipement_bancaire :
            
        st.subheader("Équipement bancaire des clients")
        
        fig_balance=plt.figure(figsize=(12,5))
        df_clean["classe_balance"]=pd.cut(df_clean.balance,bins=[-7000,0,800,2000,4000,90000],labels=['<=0',']0,800]',']800,2000]',']2000,4000]', ">4000"])
        sns.countplot(x="classe_balance",data=df_clean, edgecolor="black", alpha=0.7,hue="deposit")
        plt.title("Souscription de dépôt à terme en fonction des encours détenus par client")
        plt.xlabel("Tranche d'encours")
        plt.ylabel('Effectif')
        st.pyplot(fig_balance)
        
        st.write("Concernant les fonds sur le compte bancaire, nous pouvons voir que la grande majorité des clients ont entre 0 et 800€ sur leur compte. On remarque cependant que les personnes ayant moins de 800€ sur leur compte ont tendance à refuser de souscrire au produit. **Au-delà de 800€** la tendance s’inverse et **le taux de signature du produit devient plus important** que le taux de négatif.")
        
        
        fig_housing=plt.figure(figsize=[12,5])
        sns.countplot(x='housing', hue='deposit',edgecolor="black", alpha=0.7, data=df_clean)
        plt.title("Souscription de dépôt à terme en fonction de la détention d'un prêt immobilier")
        plt.xlabel("Détention de prêt immobilier")
        plt.ylabel('Effectif')
        st.pyplot(fig_housing)
        
        fig_loan=plt.figure(figsize=[12,5])
        sns.countplot(x='loan', hue='deposit',edgecolor="black", alpha=0.7, data=df_clean,order=("yes","no"))
        plt.title("Souscription de dépôt à terme en fonction de la détention d'un prêt personnel")
        plt.xlabel("Détention de prêt personnel")
        plt.ylabel('Effectif')
        st.pyplot(fig_loan)
        
        st.write("L’analyse graphique nous démontre que **les clients n’ayant pas de prêt immobilier sont une cible à privilégier** pour la souscription du produit. Plus de 30% des clients sondés n’ont pas de prêts immobiliers et ont souscrit au produit, tandis que près de 65% des clients ayant ce type de prêt n’ont pas souhaité adhérer au produit proposé par les conseillers bancaires.")
        st.write("On peut voir qu’une grande majorité de la population sondée ne possède pas de prêt à la consommation (+ de 85%). Le graphique nous démontre que les clients n’ayant pas souscrit à ce type de prêt n’ont pas spécialement adhéré au produit, mais n’ont pas non plus refusé l’adhésion, on peut distinguer un équilibre et en conclure que cette catégorie de la variable n’influe pas sur notre variable cible. En revanche, **les personnes ayant un prêt à la consommation refusent en grande majorité (+ de 66%)** le produit proposé par la banque.")
        st.write("Enfin, concernant les défauts de paiement : très peu de clients ont eu des défauts de paiement sur des crédits déjà en cours (1,5% de la base client) et nous estimons que cette variable n’est pas forcément très influente sur la variable cible.")

    campagne_marketing=st.checkbox("Caractéristiques de la campagne marketing")

    if campagne_marketing :
            
        st.subheader("Caractéristiques de la campagne marketing")
        
        st.write("Ces caractéristiques vont être présenté en 3 parties pour plus de lisibilité.")
        
        st.subheader("Précédente campagne")
        
        fig_lastcampaign= plt.figure(figsize=[12,5])
        sns.countplot(x='poutcome', hue='deposit',edgecolor="black", alpha=0.7, data=df_clean)
        plt.title("Souscription de dépôt à terme en fonction des résultats de la précédente campagne marketing")
        plt.xlabel("Résultat de la précédente campagne marketing")
        plt.ylabel('Effectif')
        st.pyplot(fig_lastcampaign)
        
        st.write("Si l’on regarde les résultats de la précédente campagne marketing, on remarque que les clients déjà contactés lors de la précédente campagne sont plus susceptibles de souscrire à un dépôt à terme. Et plus précisément, les clients ayant eu un succès lors de la précédente campagne marketing ont une probabilité bien plus élevée de souscrire au dépôt à terme : en effet, **91% des clients ayant souscrit lors de la précédente campagne marketing vont à nouveau souscrire lors de la campagne actuelle**. Alors que les clients qui n'ont pas été contactés lors de la précédente campagne ont un taux de souscription de seulement 41%.")
    
        st.subheader("Campagne actuelle")
        
        fig_campaign=plt.figure(figsize=[12,5])
        sns.countplot(x='campaign', hue='deposit',edgecolor="black", alpha=0.7, data=df_clean)
        plt.title("Souscription de dépôt à terme en fonction du nombre de contacts pendant la campagne")
        plt.xlabel("Nombre de contacts")
        plt.ylabel('Effectif')
        st.pyplot(fig_campaign)
        
        st.write("Si on s’intéresse à présent au nombre de contacts qui ont été effectués pendant la campagne, on remarque qu’il est préférable de ne pas trop contacter le client : en effet, **les clients qui ont été appelés une seule fois au cours de la campagne semblent avoir de meilleurs taux de souscription**.")

        fig_contact=plt.figure(figsize=[12,5])
        sns.countplot(x='contact', hue='deposit',edgecolor="black", alpha=0.7, data=df_clean)
        plt.title("Souscription de dépôt à terme en fonction du canal du dernier contact")
        plt.xlabel("Canal du dernier contact")
        plt.ylabel('Effectif')
        st.pyplot(fig_contact)
        
        st.write("Le canal du dernier contact semble également avoir une certaine influence sur la souscription car **les clients qui ont été appelés sur leur téléphone mobile ont plus souscrit** que ceux appelé sur leur téléphone fixe.")
        
        df_clean['duration_mn'] = df_clean['duration'].apply(lambda n:n/60).round(1)
        fig_duration=plt.figure(figsize=(15,5))
        df_clean["decile_duration"] = pd.qcut(df_clean["duration_mn"], q=10)
        sns.countplot(x="decile_duration",data=df_clean,edgecolor="black", alpha=0.7,hue="deposit")
        plt.xlabel("Durée de l'appel en minutes")
        plt.ylabel('Effectif')
        st.pyplot(fig_duration)
        
        st.write("Ce qui semble le plus corrélé avec la souscription au dépôt à terme est la durée du dernier appel. En effet, **plus cet appel est long, plus le client a de chance de souscrire.72% des appels qui ont duré plus de 5 mn ont donné lieu à une souscription de produit** (contre 29% des appels de moins de 5 mn). Ce qui parait logique, plus le client est intéressé par le produit, plus il se renseigne et plus l'appel va durer. A contrario, un client qui n’est pas du tout intéressé risque d’écourter l'appel.")
    
        st.subheader("Temporalité de la campagne")
        fig_month=plt.figure(figsize=(10,5))
        sns.countplot(x="month",hue="deposit",data=df_clean,edgecolor="black", alpha=0.7,order=("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"))
        plt.title('Souscription de dépôt à terme en fonction du mois du dernier contact')
        plt.xlabel("Mois du dernier contact")
        plt.ylabel('Effectif')
        st.pyplot(fig_month)
        
        st.write("Nous allons nous intéresser à l’éventuelle temporalité de la campagne. En effet, grâce au jeu de données, nous connaissons le jour et le mois du dernier appel.")
        st.write("Le graphique suivant nous montre clairement qu’il y a des moments de l’année bien plus propices que d’autres pour effectuer les appels :")
        st.write("-	le mois de mai (où il y a eu le plus d’appels) ainsi que la période estivale (de juin à août) ont été beaucoup moins bons en terme de souscription")
        st.write("-	les mois de mars, septembre et octobre ont quant à eux été les meilleurs mois")
        
        fig_calendar=plt.figure(figsize=(14,9))
        sns.swarmplot(x="month",y="day",hue="deposit",data=df_clean,order=("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"))
        plt.title('Saisonalité de la campagne de communication')
        plt.xlabel("Mois du dernier contact")
        plt.ylabel('Jour du dernier contact')
        st.pyplot(fig_calendar)
        
        st.write("On s’intéresse à présent au jour où le dernier contact a été effectué. La variable ‘day’ toute seule n’apporte pas forcément beaucoup d’informations mais si on la croise avec le mois du dernier contact, on peut obtenir un calendrier plus précis des **périodes d’appels qui ont le mieux fonctionné : de mi-février à fin mars, de septembre à la mi-novembre puis en décembre**. Et nous avons aussi la confirmation que la **période de mai à fin août serait à éviter**.")
               
             
    st.subheader("Hypothèses retenues")
        
    st.write("L’analyse graphique nous a donc permis d’identifier le profil “type” du client qui semble intéressé par la souscription d’un dépôt à terme.")
        
    st.image("fiche persona 1.jpg")
        
    st.write("Nous avons pu aussi définir le profil des clients qui semblent le moins appétents à la souscription du dépôt à terme.")
        
    st.image("fiche persona 2.jpg")
 
 
    
elif page==pages[3]:
    st.header("Préparation du jeu de données")
 
    
 
    # Nettoyage des variables quali:
    # Dataframe spécifique pour la classification
    df_classification = df_clean

    # Suppressions des colonnes créées pour la dataviz:
    df_classification.drop(['classe_age', 'classe_balance'], axis = 1, inplace = True)

    # Suppression des variables pday, previous et day
    df_classification.drop(['pdays', 'previous', 'day'], axis = 1, inplace = True)

    # Remplacement des 'no' par 0 et des yes par '1'
    df_classification.replace(['no','yes'],[0,1], inplace=True)

    # Application du One Hot Encoding sur les variables quali restantes
    df_classification_categorielles = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    for col in df_classification_categorielles :
        df_classification=df_classification.join(pd.get_dummies(df_classification[col], prefix=col).astype('int64'))

    df_classification.drop(['job', 'marital', 'education', 'contact', 'month', 'poutcome'], axis = 1, inplace = True)    


    
    st.write("Dans cette partie nous allons voir les modifications que nous avons effectuées sur le jeu de données avant de faire la modélisation. A l'issue de ces modifications, les différents algorithmes de classification ont été testés.")
    
    st.subheader("Suppression des variables non pertinentes")
    st.write("Afin de simplifier au maximum la modélisation, nous avons voulu identifié les variables explicatives qui ne présentent pas d'intérêt particulier pour l'apprentissage supervisé.")
    st.write("Ainsi, Les variables *pdays* et *previous* concernant la précédente campagne marketing ont une influence sur le résultat de *poutcome* et leur intérêt dans la prédiction de la prochaine campagne semble alors limité.*poutcome* contient l’information nécessaire et suffisante concernant la précédente campagne : c’est-à-dire le résultat final.")
    st.write("**Nous avons donc décidé de supprimer *pdays* et *previous*.**")
    st.write("Concernant la campagne de communication, nous avons pu observer une temporalité en fonction du mois de contact grâce à la dataviz tandis que l’influence du jour de contact était moins évidente. Le score des différents modèles testés avec la variable *day* n’était jamais supérieur.")
    st.write("**Nous avons également décidé de retirer cette variable de la modélisation.**")
          
    st.subheader("Encodage des variables catégorielles")
    
    st.write("Nous nous sommes également interrogés sur le traitement des variables catégorielles cardinales. En effet, les algorithmes d’apprentissage supervisé ne traitent majoritairement en entrée que des valeurs numériques.")
    st.write("Nous avons donc eu recours au **one hot encoding**. Cette méthode a permis de transformer chaque valeur distincte des colonnes catégorielles en une nouvelle colonne qui prend en valeur 0 ou 1.")
    st.write("Les variables concernées par cet encodage sont: ")
    st.write(" - job")
    st.write(" - marital")
    st.write(" - education")
    st.write(" - contact")
    st.write(" - month")
    st.write(" - poutcome")
    
    st.subheader("Normalisation des variables quantitatives")
    
    st.write("Après avoir supprimé les variables non pertinentes et encodé les variables catégorielles, nous devions normaliser les variables quantitatives.")  
    st.write("L’objectif de la normalisation est de centrer et réduire toutes les valeurs des variables numériques autour de zéro afin d’utiliser une échelle commune, sans que les différences de plages de valeurs ne soient faussées et sans perte d'informations. Cette technique est indispensable pour le bon fonctionnement d’algorithmes tels que la régression logistique, les SVM (machines à vecteurs de support) ou encore la méthode des KNN (méthode des K plus proches voisins). Cette étape est réalisée après la séparation du jeu de données.")
    st.write("Les variables numériques suivantes ont donc été normalisées:")
    st.write(" - age")
    st.write(" - balance")
    st.write(" - duration")
    st.write(" - campaign") 
            
    if st.checkbox("Afficher un aperçu du jeu de données modifié (avant normalisation)"):
        st.dataframe(df_classification)



elif page==pages[4]:
    st.header("Modélisation")
    st.write("Pour procéder à l’apprentissage supervisé, les données ont été séparées en deux : d'une part les variables explicatives, d'autre part la variable à prédire (deposit).")
    st.write("La base de données a également été scindée en deux afin d’avoir un premier ensemble d’entraînement et un deuxième ensemble de test.")
    
    # Nettoyage des variables quali:
    # Dataframe spécifique pour la classification
    df_classification = df_clean

    # Suppressions des colonnes créées pour la dataviz:
    df_classification.drop(['classe_age', 'classe_balance'], axis = 1, inplace = True)

    # Suppression des variables pday, previous et day
    df_classification.drop(['pdays', 'previous', 'day'], axis = 1, inplace = True)

    # Remplacement des 'no' par 0 et des yes par '1'
    df_classification.replace(['no','yes'],[0,1], inplace=True)

    # Application du One Hot Encoding sur les variables quali restantes
    df_classification_categorielles = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    for col in df_classification_categorielles :
        df_classification=df_classification.join(pd.get_dummies(df_classification[col], prefix=col).astype('int64'))

    df_classification.drop(['job', 'marital', 'education', 'contact', 'month', 'poutcome'], axis = 1, inplace = True)    
    
    feats, target = df_classification.drop('deposit', axis=1), df_classification['deposit']

    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state = 100)
    
    quanti = ['age', 'balance', 'duration', 'campaign']

    sc = StandardScaler()
    X_train[quanti] = sc.fit_transform(X_train[quanti])
    X_test[quanti] = sc.transform(X_test[quanti])
    
    if st.checkbox("Afficher un aperçu du set d'entraînement normalisé"):
        st.dataframe(X_train)    
        
    st.subheader("Modèles de classification testés")
    
    # Chargement des modèles
    cl_LR = load('cl_LR.joblib') 
    cl_SVM = load('cl_SVM.joblib') 
    cl_KNN = load('cl_KNN.joblib') 
    cl_DTC = load('cl_DTC.joblib') 
    cl_RFC = load('cl_RFC.joblib') 
    cl_AC = load('cl_AC.joblib') 
    cl_BC = load('cl_BC.joblib') 

    cl_LR_final = load('cl_LR_final.joblib') 
    cl_RFC_final = load('cl_RFC_final.joblib') 

    model_choisi = st.selectbox(label = "Choix du modèle", options = ['Régression Logistique', 'SVM', 'KNN', 'Decision Tree', 'Forêt aléatoire', 'Adaboost', 'Bagging'])
    def test_model(model_choisi): 
        if model_choisi == 'Régression Logistique':
            model = cl_LR
        elif model_choisi == 'SVM':
            model = cl_SVM
        elif model_choisi == 'KNN':
            model = cl_KNN
        elif model_choisi == 'Decision Tree': 
            model = cl_DTC
        elif model_choisi == 'Forêt aléatoire':
            model = cl_RFC
        elif model_choisi == 'Adaboost':
            model = cl_AC
        elif model_choisi == 'Bagging':
            model = cl_BC
        score = model.score(X_test,y_test)
        return score

    def avantages_model(model_choisi): 
        if model_choisi == 'Régression Logistique':
            avantages = st.markdown("<font color='green'>Avantages : Score supérieur à 0.8, interprétabilité excellente, temps d’entraînement rapide</font>", unsafe_allow_html=True)
        elif model_choisi == 'SVM':
            avantages = st.markdown("<font color='green'>Avantages : Score supérieur à 0.8</font>", unsafe_allow_html=True)
        elif model_choisi == 'KNN':
            avantages = ""
        elif model_choisi == 'Decision Tree': 
            avantages = ""
        elif model_choisi == 'Forêt aléatoire':
            avantages = st.markdown("<font color='green'>Avantages : Score le plus élevé, supérieur à 0.8</font>", unsafe_allow_html=True)
        elif model_choisi == 'Adaboost':
            avantages = ""
        elif model_choisi == 'Bagging':
            avantages = st.markdown("<font color='green'>Avantages : Score supérieur à 0.8</font>", unsafe_allow_html=True)
        return avantages

    def inconvenients_model(model_choisi): 
        if model_choisi == 'Régression Logistique':
            inconvenients = st.markdown("<font color='red'>Inconvénients : Score légèrement inférieur à d'autres modèles comme la Random Forest</font>", unsafe_allow_html=True)
        elif model_choisi == 'SVM':
            inconvenients = st.markdown("<font color='red'>Inconvénients : difficulté à identifier les bonnes valeurs des paramètres, difficulté d’interprétations (ex. pertinence des variables), temps d’entraînement excessivement long</font>", unsafe_allow_html=True)
        elif model_choisi == 'KNN':
            inconvenients = st.markdown("<font color='red'>Inconvénients : Score inférieur à 0.8 (trop faible par rapport aux autres modèles)</font>", unsafe_allow_html=True)
        elif model_choisi == 'Decision Tree': 
            inconvenients = st.markdown("<font color='red'>Inconvénients : Score inférieur à 0.8 (trop faible par rapport aux autres modèles)</font>", unsafe_allow_html=True)
        elif model_choisi == 'Forêt aléatoire':
            inconvenients = st.markdown("<font color='red'>Inconvénients : Interprétabilité moins bonne qu'une régression logistique</font>", unsafe_allow_html=True)
        elif model_choisi == 'Adaboost':
            inconvenients = st.markdown("<font color='red'>Inconvénients : Score inférieur à 0.8 (trop faible par rapport aux autres modèles)</font>", unsafe_allow_html=True)
        elif model_choisi == 'Bagging':
            inconvenients = st.markdown("<font color='red'>Inconvénients : Score qui ne dépasse pas un modèle similaire et légèrement plus rapide : le Random Forest</font>", unsafe_allow_html=True)
        return inconvenients
      
    st.write("Test score", test_model(model_choisi))
    avantages_model(model_choisi)
    inconvenients_model(model_choisi)
    
    st.subheader("Présentation des modèles retenus pour la banque")
        
    st.markdown("##### Régression logistique")
    
    if st.checkbox("La matrice de confusion"):
        st.image('confusion_matrix_cl_lr.png')         
    if st.checkbox("Les métriques de performance"):
        st.write("- **Rappel de la classe positive = 78%** soit environ 4/5 de bonnes prédictions pour les clients ayant accepté le produit bancaire")
        st.write("- *Rappel de la classe négative = 87%* soit près de 9/10 bonnes prédictions pour les clients ayant refusé le produit bancaire")
        st.write("- **Absence d'overfitting** :")
        st.write("Test score", cl_LR_final.score(X_test,y_test))
        st.write("Train score", cl_LR_final.score(X_train,y_train))
    if st.checkbox("Les coefficients"):
        st.image('coef_lr.png')  
    if st.checkbox("Les indicateurs clés du modèle"):
        st.write("- **Acceptation du produit bancaire :** Les étudiants et les retraités, ayant reçu un contact : telephone ou cellular d'une longue durée pour les mois de Mars, Septembre, Octobre ou Décembre ainsi que les personnes ayant déjà accepté un précédent dépôt bancaire (poutcome = success).")
        st.write("- **Refus du produit bancaire :** Les personnes ayant un crédit immobilier ou un emprunt bancaire, ayant refusé le précédent dépôt bancaire ou n'ayant pas été contacté précédemment et ayant été contacté pour cette campagne par un autre type de contact (other than telephone or cellular) en Janvier, Mai, Juillet, Août ou Novembre.")
    
    st.markdown("##### Forêt aléatoire")
        
    if st.checkbox("La matrice de confusion "):
        st.image('confusion_matrix_cl_RFC.png') 
    if st.checkbox("Les métriques de performance "):
        st.write("- **Rappel de la classe positive = 88%** soit près de 9/10 bonnes prédictions pour les clients ayant accepté le produit bancaire")
        st.write("- *Rappel de la classe négative = 83%* soit plus de 8/10 bonnes prédictions pour les clients ayant refusé le produit bancaire")
        st.write("- **Faible présence d’overfitting** :")
        st.write("Test score", cl_RFC_final.score(X_test,y_test))
        st.write("Train score", cl_RFC_final.score(X_train,y_train))
    if st.checkbox("Les coefficients "):
        st.image('coef_rf.png') 
    if st.checkbox("Les indicateurs clés du modèle "):
        st.write("- Pour ce modèle, les variables clés de l’algorithme sont la durée (en secondes) du dernier contact de la campagne en cours, le résultat de la campagne marketing précédente et plus particulièrement si cela a été un succès, le montant des encours détenus par le client, l’âge, le canal de communication (lors du dernier contact de la campagne en cours) et la détention ou non d'un prêt immobilier.")



elif page==pages[5]:
    st.header("Conclusion")
    
    
    st.subheader("A. Recommandations pour les prochaines campagnes")
    
    st.write("Suite à l’analyse via la dataviz et la modélisation, nous pouvons faire plusieurs recommandations pour les prochaines campagnes. En effet, nous pouvons déterminer des clients à cibler et certaines caractéristiques de la campagne.")
    
    st.markdown("##### Clients à cibler :")
    st.write("Notre cible prioritaire devra se concentrer sur les clients qui ont moins de 30 ans ou plus de 60 ans, étudiants ou retraités, n’ayant pas de prêt en cours et avec un solde bancaire de plus de 800€. Il faudra appeler en priorité les clients qui ont déjà été ciblés lors d’une précédente campagne ou encore mieux (si les volumes sont suffisants), appeler ceux qui ont déjà souscrit au produit lors d’une précédente campagne.")
    
    st.markdown("##### Caractéristiques de la campagne :")
    st.write("De plus, pour une meilleure réussite, la campagne devra se dérouler durant les mois de mars, septembre, octobre et décembre (éviter le mois de mai et la période estivale).")
    st.write("Le contact devra se faire sur le téléphone mobile. Par ailleurs, il serait judicieux d’inciter les téléconseillers à faire durer l’appel téléphonique car plus celui-ci sera long, plus le client sera enclin à adhérer au produit.")
    
    
    st.subheader("B. Choix de l’algorithme") 
    
    st.write("Nous avons décidé de proposer 2 modèles différents à la banque qui décidera en fonction de ses objectifs :")
    st.write("- le modèle de **régression logistique** qui présente de bonnes performances, rapide à exécuter et une très bonne interprétabilité du modèle grâce aux coefficients de la régression. Cependant, le nombre de “faux négatifs” est un peu élevé, ce qui signifie qu’avec ce modèle la banque pourrait passer à côté de certains clients intéressés par le dépôt à terme et qui n’ont pas été détectés par le modèle.")
    st.write("- le modèle de **forêt aléatoire** qui présente de meilleures performances notamment avec un meilleur rappel (grâce à un nombre de “faux négatifs” moins élevé). Mais le modèle est un peu plus long à l’exécution et surtout, il est plus difficile à interpréter.")
    
    
    st.subheader("C. Bilan du projet")
    
    st.write("Lors du projet nous avons rencontré quelques difficultés notamment en termes d’ajustement des hyperparamètres des modèles, en termes de temps d’exécution des algorithmes, mais aussi par rapport à l’interprétabilité de la variable duration que nous avons hésité à conserver.")
    st.write("Concernant la répartition des tâches, nous avons tous réalisé les différentes étapes du projet (analyse rapide des variables et du jeu de données, dataviz, modélisations), puis nous avons partagé nos retours sur ces étapes. Ensuite un membre de l’équipe synthétisait nos remarques et conclusions, et les appliquait à l’étape du projet.")
    st.write("Ce projet nous a permis de nous rendre compte des problématiques des entreprises du secteur bancaire. Par ailleurs, nous avons pu mettre en pratique plusieurs parties du cours et avoir ainsi une première vision des difficultés et facilités du métier de data analyst.")

    