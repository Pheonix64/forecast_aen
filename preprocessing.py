import pandas as pd
import os

def transform_data_dynamic(df, row_index_data, name_column):
    """
    Transforme un DataFrame en un format de série temporelle en se basant sur la ligne de données spécifiée.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée sans en-tête.
        row_index_data (int): L'index de la ligne qui contient les données à transformer.
        name_column (str): Le nom souhaité pour la colonne des valeurs.

    Returns:
        pd.DataFrame: Un DataFrame transformé avec les colonnes 'Date' et la colonne des valeurs.
    """
    if row_index_data >= len(df):
        raise IndexError(f"L'index de ligne {row_index_data} est hors des limites du DataFrame. Le nombre maximum de lignes est {len(df) - 1}.")
    
    # Sélectionner les dates dans la première ligne
    dates_row = df.iloc[0]
    
    # Sélectionner les valeurs de la ligne spécifiée par l'utilisateur
    data_row = df.iloc[row_index_data]
    
    # Créer le DataFrame final
    transformed_df = pd.DataFrame({
        'Date': dates_row,
        name_column: data_row
    }).reset_index(drop=True)
    
    # Exclure les deux premières cellules qui ne sont pas des dates ou des valeurs
    transformed_df = transformed_df.iloc[2:].copy()
    
    # Convertir les valeurs en numérique
    transformed_df[name_column] = pd.to_numeric(transformed_df[name_column], errors='coerce')

    # Convertir les dates, on garde la logique de remplacement des mois
    month_map = {
        'JAN': '01', 'FEV': '02', 'MAR': '03', 'AVR': '04', 'MAY': '05', 'JUN': '06',
        'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
    }
    
    df_dates = transformed_df['Date'].astype(str).str.upper()
    for month_name, month_num in month_map.items():
        df_dates = df_dates.str.replace(month_name, month_num, regex=False)

    transformed_df['Date'] = pd.to_datetime(df_dates.str[-4:] + '-' + df_dates.str[:2] + '-01', errors='coerce')

    # Nettoyage et tri
    transformed_df.dropna(subset=['Date', name_column], inplace=True)
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
            
            # Afficher les 5 premières lignes pour aider l'utilisateur
            print("\nVoici un aperçu des 5 premières lignes du fichier :")
            print(df.head())
            
            # Demander à l'utilisateur de spécifier les lignes
            row_index_dates = int(input("\nEntrez le numéro de la ligne qui contient les dates (commence à 0) : "))
            row_index_data = int(input("Entrez le numéro de la ligne qui contient les valeurs à extraire : "))
            name_column = input("Entrez le nom souhaité pour la colonne des valeurs (par exemple, 'Taux de change') : ")
            
            # Appeler la fonction de transformation
            # On utilise le code précédent mais on passe l'index des lignes
            # et le nom de la colonne
            # Remarque: Je me suis basé sur un format où les dates sont sur la ligne 0
            # et les données sur une autre ligne. Le code est adapté pour cette structure.
            transformed_df = transform_data_dynamic(df, row_index_data, name_column)
            
            output_file = f"transformed_data_{name_column}.xlsx"
            transformed_df.to_excel(output_file, index=False)
            
            print("\nTransformation réussie ! 🎉")
            print(f"Les données transformées ont été sauvegardées dans '{output_file}'.")
            print("\nAperçu des données transformées :")
            print(transformed_df.head())
            break
        
        except ValueError as ve:
            print(f"Erreur de saisie : {ve}")
        except FileNotFoundError:
            print(f"Erreur : Le fichier '{file_path}' n'a pas été trouvé. Vérifiez le chemin et réessayez.")
        except Exception as e:
            print(f"Une erreur inattendue est survenue : {e}")