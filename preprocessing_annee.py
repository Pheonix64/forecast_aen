import pandas as pd
import os
import re

def transform_data_dynamic(df, data_label, date_row_index=0):
    """
    Transforme un DataFrame en série temporelle en se basant sur un libellé de données
    et un index de ligne fixe pour les dates.
    
    Args:
        df (pd.DataFrame): Le DataFrame d'entrée sans en-tête.
        data_label (str): Le libellé de la ligne contenant les valeurs (par exemple, "PIB nominal").
        date_row_index (int): L'index de la ligne contenant les dates (par défaut à 0).

    Returns:
        pd.DataFrame: Un DataFrame transformé avec les colonnes 'Date' et 'Valeur'.
    """
    # Chercher la ligne de données par son libellé
    data_row_series = df[df.iloc[:, 1].str.contains(re.escape(data_label), case=False, na=False)].iloc[0]

    # Utiliser la ligne d'en-tête (index 0) pour les dates
    dates_row_series = df.iloc[date_row_index]

    # Créer le DataFrame final en combinant les deux séries
    transformed_df = pd.DataFrame({
        'Date': dates_row_series,
        'Valeur': data_row_series
    })
    
    # Supprimer les deux premières colonnes non pertinentes
    transformed_df = transformed_df.iloc[2:]

    # Convertir les dates en années et les valeurs en numérique
    transformed_df['Date'] = pd.to_numeric(transformed_df['Date'], errors='coerce')
    transformed_df['Date'] = pd.to_datetime(transformed_df['Date'], format='%Y', errors='coerce')
    transformed_df['Valeur'] = pd.to_numeric(transformed_df['Valeur'], errors='coerce')
    
    # Nettoyer et trier les données
    transformed_df.dropna(subset=['Date', 'Valeur'], inplace=True)
    transformed_df.sort_values(by='Date', inplace=True)
    
    return transformed_df.reset_index(drop=True)

if __name__ == "__main__":
    print("Bienvenue dans le programme de transformation de données ! 😊")
    
    while True:
        file_path = input("Veuillez entrer le chemin du fichier Excel (.xlsx ou .xls) : ")
        if not os.path.exists(file_path):
            print(f"Erreur : Le fichier '{file_path}' n'a pas été trouvé. Veuillez réessayer.")
            continue
            
        try:
            # Lire le fichier sans en-tête
            df = pd.read_excel(file_path, header=None)
            
            # Afficher un aperçu pour aider l'utilisateur
            print("\nVoici un aperçu des 10 premières lignes du fichier :")
            print(df.head(10).to_string())
            
            # Demander le libellé de la ligne de données
            data_label = input("Entrez le libellé de la ligne qui contient les VALEURS à extraire (par exemple, 'PIB nominal') : ")
            
            # Appeler la fonction de transformation
            transformed_df = transform_data_dynamic(df, data_label)
            
            output_file = f"transformed_data_{data_label.replace(' ', '_')}.xlsx"
            transformed_df.to_excel(output_file, index=False)
            
            print("\nTransformation réussie ! 🎉")
            print(f"Les données transformées ont été sauvegardées dans '{output_file}'.")
            print("\nAperçu des données transformées :")
            print(transformed_df.head().to_string())
            break
        
        except ValueError as ve:
            print(f"Erreur de saisie : {ve}")
        except FileNotFoundError:
            print(f"Erreur : Le fichier '{file_path}' n'a pas été trouvé. Vérifiez le chemin et réessayez.")
        except IndexError:
            print("Erreur : Le libellé que vous avez entré n'a pas été trouvé. Veuillez vérifier l'orthographe ou le bon libellé dans le fichier.")
        except Exception as e:
            print(f"Une erreur inattendue est survenue : {e}")