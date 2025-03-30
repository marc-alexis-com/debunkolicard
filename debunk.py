#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script amélioré pour trouver des chemins dans Wikipédia qui épellent un mot
à partir des premières lettres des titres des pages.

Dépendances requises:
- requests
- beautifulsoup4
- tqdm
- networkx
- matplotlib

Dépendances optionnelles:
- seaborn (pour des visualisations améliorées)
- pandas (pour des analyses de données)
"""

import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, unquote
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from concurrent.futures import ThreadPoolExecutor
import json
import os
from tqdm import tqdm
import argparse
from collections import Counter
import numpy as np
import random

# Importation des packages optionnels avec gestion d'erreurs
try:
    import pandas as pd
except ImportError:
    pd = None
    print("Warning: pandas n'est pas installé. Certaines fonctionnalités statistiques seront limitées.")

try:
    import seaborn as sns
except ImportError:
    sns = None
    print("Warning: seaborn n'est pas installé. Certaines visualisations seront simplifiées.")


class WikipathFinder:
    def __init__(self, start_url, target_word='EFREI', max_paths=10, variants_per_letter=None,
                 links_per_level=3, cache_file='wiki_cache.json', output_dir='results'):
        self.start_url = start_url
        self.target_word = target_word.upper()
        self.max_paths = max_paths
        self.links_per_level = links_per_level

        # Nombre de variantes spécifiques par lettre (dictionnaire)
        if variants_per_letter is None:
            # Par défaut, utiliser le même nombre pour toutes les lettres sauf I
            self.variants_per_letter = {
                letter: links_per_level for letter in self.target_word}
            # Pour I, utiliser 5 variantes
            if 'I' in self.target_word:
                self.variants_per_letter['I'] = 5
        else:
            self.variants_per_letter = variants_per_letter

        self.cache_file = cache_file
        self.output_dir = output_dir
        self.cache = self._load_cache()
        self.all_paths = []
        self.stats = {
            'pages_visited': set(),
            'total_links_found': 0,
            'links_per_letter': {letter: [] for letter in self.target_word},
            'time_per_level': [],
            'success_rate_per_letter': {}
        }

        # Crée le dossier de sortie s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _load_cache(self):
        """Charge le cache depuis un fichier s'il existe."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Sauvegarde le cache dans un fichier."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def get_wikipedia_links(self, url):
        """Récupère tous les liens d'une page Wikipédia avec mise en cache."""
        # Vérifie si l'URL est dans le cache
        if url in self.cache:
            return self.cache[url]

        try:
            self.stats['pages_visited'].add(url)
            # Ajoute un user agent pour éviter d'être bloqué
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Recherche tous les liens dans le contenu principal
            content_div = soup.find(id="mw-content-text")
            if not content_div:
                self.cache[url] = []
                return []

            links = []
            for a_tag in content_div.find_all('a', href=True):
                href = a_tag['href']
                # Ne considère que les liens internes de Wikipédia vers des articles
                if href.startswith('/wiki/') and ':' not in href and 'File:' not in href and 'Special:' not in href:
                    full_url = urljoin('https://fr.wikipedia.org', href)
                    # Extrait le titre de l'URL
                    title = unquote(href.split('/')[-1].replace('_', ' '))
                    links.append((title, full_url))

            # Met à jour les statistiques
            self.stats['total_links_found'] += len(links)

            # Ajoute au cache et respecte les serveurs de Wikipédia
            self.cache[url] = links
            self._save_cache()
            time.sleep(0.5)
            return links
        except Exception as e:
            print(f"Erreur lors de la récupération de {url}: {e}")
            self.cache[url] = []
            return []

    def filter_links_by_first_letter(self, links, letter):
        """Filtre les liens pour ne garder que ceux dont le titre commence par la lettre donnée."""
        filtered = [(title, url) for title,
                    url in links if title and title[0].upper() == letter.upper()]

        # Met à jour les statistiques pour cette lettre
        if filtered:
            self.stats['links_per_letter'][letter].extend(
                [title for title, _ in filtered])

        return filtered

    def find_paths(self, start_title="Banane"):
        """Trouve des chemins à partir de l'URL de départ qui épellent le mot cible."""
        self.all_paths = []

        # Récupère les liens de la page de départ
        print(f"Récupération des liens depuis {self.start_url}")
        start_time = time.time()
        start_links = self.get_wikipedia_links(self.start_url)

        # Initialise un chemin vide
        current_path = [(start_title, self.start_url)]

        # Lance la recherche pour la première lettre (E pour EFREI)
        first_letter = self.target_word[0]
        e_links = self.filter_links_by_first_letter(start_links, first_letter)
        print(f"Trouvé {len(e_links)} liens commençant par {first_letter}")

        # Pour chaque variante de la première lettre (E)
        variants_for_current_level = min(
            len(e_links), self.variants_per_letter.get(first_letter, self.links_per_level))

        for e_idx, (e_title, e_url) in enumerate(e_links[:variants_for_current_level]):
            print(
                f"[{e_idx+1}/{variants_for_current_level}] Exploration du lien {first_letter}: {e_title}")

            # Crée un chemin avec cette variante
            new_path = current_path.copy()
            new_path.append((e_title, e_url))

            # Continue avec le niveau suivant (F, R, E, I)
            self._search_next_level(new_path, 1)

        # Enregistre le temps total
        end_time = time.time()
        self.stats['total_time'] = end_time - start_time

        # Calcule le taux de réussite par lettre
        for letter in self.target_word:
            total_attempts = len(self.stats['links_per_letter'][letter])
            if total_attempts > 0:
                # Compte combien de fois chaque lien a été utilisé dans les chemins réussis
                successful_links = []
                for path in self.all_paths:
                    for i, (title, _) in enumerate(path):
                        if i > 0 and i <= len(self.target_word) and self.target_word[i-1] == letter:
                            successful_links.append(title)

                success_count = len(successful_links)
                self.stats['success_rate_per_letter'][letter] = success_count / \
                    total_attempts
            else:
                self.stats['success_rate_per_letter'][letter] = 0

        return self.all_paths

    def _search_next_level(self, current_path, level):
        """Recherche récursivement les liens pour compléter le mot."""
        # Si nous avons atteint le nombre maximum de chemins, arrêtez
        if len(self.all_paths) >= self.max_paths:
            return

        # Si nous avons complété le mot, ajoutez le chemin à la liste
        if level >= len(self.target_word):
            self.all_paths.append(current_path.copy())
            print(
                f"Chemin complet trouvé: {' -> '.join(title for title, _ in current_path)}")
            return

        # Récupère la lettre cible pour ce niveau
        target_letter = self.target_word[level]

        # Enregistre l'heure de début pour ce niveau
        level_start_time = time.time()

        # Récupère tous les liens de la dernière page du chemin
        _, last_url = current_path[-1]
        next_links = self.get_wikipedia_links(last_url)

        # Filtre les liens par la lettre cible
        filtered_links = self.filter_links_by_first_letter(
            next_links, target_letter)

        # Si aucun lien n'est trouvé pour cette lettre, abandonne ce chemin
        if not filtered_links:
            return

        # Nombre de variantes à explorer pour ce niveau
        variants_for_current_level = min(len(filtered_links), self.variants_per_letter.get(
            target_letter, self.links_per_level))

        # Explore les premiers liens_per_level liens
        for idx, (title, url) in enumerate(filtered_links[:variants_for_current_level]):
            # Vérifie si ce lien est déjà dans le chemin (évite les cycles)
            if any(url == path_url for _, path_url in current_path):
                continue

            # Ajoute ce lien au chemin
            current_path.append((title, url))

            # Continue avec le niveau suivant
            self._search_next_level(current_path, level + 1)

            # Backtracking: retire ce lien pour essayer d'autres possibilités
            current_path.pop()

        # Enregistre le temps pour ce niveau
        level_end_time = time.time()
        self.stats['time_per_level'].append(level_end_time - level_start_time)

    def print_paths(self):
        """Affiche les chemins dans un format lisible."""
        for i, path in enumerate(self.all_paths, 1):
            print(f"Chemin {i}:")
            for j, (title, url) in enumerate(path):
                if j == 0:
                    print(f"  Départ: {title} - {url}")
                else:
                    letter = self.target_word[j-1] if j - \
                        1 < len(self.target_word) else "?"
                    print(f"  {letter}: {title} - {url}")
            print()

    def save_paths_to_file(self, filename=None):
        """Enregistre les chemins trouvés dans un fichier texte."""
        if filename is None:
            filename = os.path.join(
                self.output_dir, f"{self.target_word}_paths.txt")

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Recherche de chemins pour épeler '{self.target_word}'\n")
            f.write(f"Point de départ: {self.start_url}\n\n")

            f.write("=== STATISTIQUES ===\n")
            f.write(
                f"Nombre de pages visitées: {len(self.stats['pages_visited'])}\n")
            f.write(
                f"Nombre total de liens trouvés: {self.stats['total_links_found']}\n")
            f.write(
                f"Temps total de recherche: {self.stats['total_time']:.2f} secondes\n\n")

            f.write("Taux de réussite par lettre:\n")
            for letter, rate in self.stats['success_rate_per_letter'].items():
                f.write(f"  {letter}: {rate*100:.1f}%\n")
            f.write("\n")

            for i, path in enumerate(self.all_paths, 1):
                f.write(f"Chemin {i}:\n")
                for j, (title, url) in enumerate(path):
                    if j == 0:
                        f.write(f"  Départ: {title} - {url}\n")
                    else:
                        letter = self.target_word[j-1] if j - \
                            1 < len(self.target_word) else "?"
                        f.write(f"  {letter}: {title} - {url}\n")
                f.write("\n")
        print(f"Chemins enregistrés dans {filename}")

    def create_visualizations(self):
        """Crée diverses visualisations statistiques."""
        if not self.all_paths:
            print("Aucun chemin à visualiser.")
            return

        # Crée le dossier de visualisations s'il n'existe pas
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        # 1. Graphe de réseau des chemins
        self._create_network_graph(os.path.join(
            viz_dir, f"{self.target_word}_network.png"))

        # 2. Histogramme des titres les plus fréquents par lettre
        self._create_title_histograms(viz_dir)

        # 3. Graphique des temps par niveau
        self._create_time_chart(os.path.join(
            viz_dir, f"{self.target_word}_time_per_level.png"))

        # 4. Taux de réussite par lettre
        self._create_success_rate_chart(os.path.join(
            viz_dir, f"{self.target_word}_success_rate.png"))

        # 5. Sunburst chart des chemins
        self._create_sunburst_chart(os.path.join(
            viz_dir, f"{self.target_word}_sunburst.png"))

        # 6. Crée une visualisation HTML interactive
        self.create_html_visualization(os.path.join(
            self.output_dir, f"{self.target_word}_visualization.html"))

    def _create_network_graph(self, filename):
        """Crée un graphe de réseau des chemins trouvés."""
        G = nx.DiGraph()

        try:
            # Utilise un jeu de couleurs harmonieux
            path_colors = plt.cm.viridis(
                np.linspace(0, 1, len(self.all_paths)))

            # Ajoute des nœuds et des arêtes pour chaque chemin
            edge_colors = []
            edges = []
            node_sizes = {}
            node_levels = {}  # Pour stocker le niveau de chaque nœud

            for i, path in enumerate(self.all_paths):
                # Utilise une couleur du jeu de couleurs
                color = path_colors[i]

                # Ajoute chaque nœud du chemin
                for j, (title, _) in enumerate(path):
                    # Crée un label spécifique au niveau
                    if j == 0:
                        node_label = f"Départ: {title}"
                        level = 0
                    else:
                        node_label = f"{self.target_word[j-1]}: {title}"
                        level = j

                    # Ajoute le nœud s'il n'existe pas encore
                    if not G.has_node(node_label):
                        G.add_node(node_label)
                        node_sizes[node_label] = 300  # Taille de base
                        node_levels[node_label] = level
                    else:
                        # Augmente la taille du nœud s'il est utilisé dans plusieurs chemins
                        node_sizes[node_label] += 150

                    # Ajoute une arête au nœud précédent
                    if j > 0:
                        prev_title = path[j-1][0]
                        prev_label = f"{'Départ' if j-1 == 0 else self.target_word[j-2]}: {prev_title}"
                        edges.append((prev_label, node_label))
                        edge_colors.append(color)

            # Ajoute toutes les arêtes au graphe
            G.add_edges_from(edges)

            # Crée la figure avec un fond blanc
            plt.figure(figsize=(15, 12), facecolor='white')

            # Utilise un layout hiérarchique pour une meilleure visualisation des niveaux
            try:
                pos = nx.multipartite_layout(
                    G, subset_key=lambda node: node_levels.get(node, 0))
            except:
                # Fallback sur spring layout si multipartite layout échoue
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

            # Utilise différentes couleurs pour les nœuds selon leur niveau
            node_colors = [plt.cm.tab10(node_levels.get(
                node, 0) % 10) for node in G.nodes()]

            # Dessine les nœuds avec des tailles variables
            nx.draw_networkx_nodes(G, pos,
                                   node_size=[node_sizes[node]
                                              for node in G.nodes()],
                                   node_color=node_colors,
                                   alpha=0.8,
                                   edgecolors='white')

            # Dessine les arêtes colorées par chemin
            for i, (edge, color) in enumerate(zip(edges, edge_colors)):
                nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2.0, alpha=0.7,
                                       edge_color=[color], arrows=True, arrowsize=15)

            # Ajoute les étiquettes des nœuds avec un léger décalage
            label_pos = {k: (v[0], v[1] + 0.03) for k, v in pos.items()}
            nx.draw_networkx_labels(G, label_pos, font_size=9, font_family='sans-serif',
                                    font_weight='bold', horizontalalignment='center')

            # Ajoute un titre et des informations
            plt.title(
                f"Réseau des chemins Wikipédia pour '{self.target_word}'", fontsize=18, pad=20)
            plt.text(0.5, -0.05, f"Nombre de chemins: {len(self.all_paths)} | Pages visitées: {len(self.stats['pages_visited'])}",
                     horizontalalignment='center', transform=plt.gca().transAxes, fontsize=12)

            # Ajoute une légende pour les niveaux
            legend_elements = []
            level_names = ["Départ"] + list(self.target_word)
            for i, name in enumerate(level_names[:len(set(node_levels.values()))]):
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i % 10),
                               markersize=10, label=f"Niveau {name}")
                )
            plt.legend(handles=legend_elements,
                       loc='upper left', bbox_to_anchor=(1, 1))

            # Retire les axes
            plt.axis('off')

            # Ajoute une marge pour la lisibilité
            plt.tight_layout(pad=1.2)

            # Sauvegarde la figure
            plt.savefig(filename, format="PNG", dpi=300, bbox_inches='tight')
            print(f"Graphe réseau sauvegardé dans {filename}")
            plt.close()

        except Exception as e:
            print(f"Erreur lors de la création du graphe réseau: {e}")
            # Essaie avec un graphe plus simple en cas d'erreur
            try:
                plt.figure(figsize=(15, 12))
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500,
                        font_size=8, font_weight='bold', edge_color='gray', arrows=True)
                plt.title(
                    f"Chemins Wikipédia pour '{self.target_word}'", fontsize=18)
                plt.savefig(filename, format="PNG", dpi=300)
                print(f"Graphe réseau simplifié sauvegardé dans {filename}")
                plt.close()
            except Exception as e2:
                print(f"Échec de la création du graphe réseau simplifié: {e2}")

    def _create_title_histograms(self, viz_dir):
        """Crée des histogrammes des titres les plus fréquents pour chaque lettre."""
        for letter in self.target_word:
            try:
                titles = self.stats['links_per_letter'][letter]
                if not titles:
                    continue

                title_counts = Counter(titles)
                top_titles = title_counts.most_common(10)

                # Prépare les données pour le graphique
                labels = [
                    title[:30] + '...' if len(title) > 30 else title for title, _ in top_titles]
                values = [count for _, count in top_titles]

                # Crée la figure
                plt.figure(figsize=(12, 6), facecolor='white')

                # Utilise seaborn si disponible pour un meilleur rendu
                if sns is not None:
                    # Configure le style seaborn
                    sns.set_style("whitegrid")
                    # Crée un graphique à barres horizontal
                    ax = sns.barplot(x=values, y=labels, palette="Blues_d")
                    # Ajoute les valeurs sur les barres
                    for i, v in enumerate(values):
                        ax.text(v + 0.1, i, f"{v:.0f}",
                                va='center', fontsize=10)
                else:
                    # Crée un graphique à barres horizontal avec matplotlib
                    bars = plt.barh(labels, values, color=plt.cm.Blues(
                        np.linspace(0.5, 0.9, len(labels))))
                    # Ajoute les valeurs sur les barres
                    for bar in bars:
                        width = bar.get_width()
                        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.0f}",
                                 ha='left', va='center', fontsize=10)

                # Ajoute un titre et des étiquettes
                plt.title(
                    f"Top 10 des titres commençant par '{letter}'", fontsize=15)
                plt.xlabel("Fréquence", fontsize=12)
                plt.ylabel("Titre", fontsize=12)

                # Ajoute une grille horizontale légère
                plt.grid(axis='x', linestyle='--', alpha=0.3)

                # Ajuste la mise en page
                plt.tight_layout()

                # Sauvegarde la figure
                filename = os.path.join(
                    viz_dir, f"{self.target_word}_top_titles_{letter}.png")
                plt.savefig(filename, format="PNG", dpi=300)
                print(
                    f"Histogramme pour la lettre {letter} sauvegardé dans {filename}")
                plt.close()

            except Exception as e:
                print(
                    f"Erreur lors de la création de l'histogramme pour la lettre {letter}: {e}")

    def _create_time_chart(self, filename):
        """Crée un graphique montrant le temps passé à chaque niveau."""
        levels = [f"{self.target_word[i]}" for i in range(
            len(self.stats['time_per_level']))]
        times = self.stats['time_per_level']

        plt.figure(figsize=(10, 6), facecolor='white')

        # Crée un graphique à barres avec une palette de couleurs dégradée
        bars = plt.bar(levels, times, color=plt.cm.viridis(
            np.linspace(0.2, 0.8, len(levels))))

        # Ajoute les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f"{height:.1f}s",
                     ha='center', va='bottom', fontsize=10)

        # Ajoute un titre et des étiquettes
        plt.title("Temps passé à explorer chaque niveau", fontsize=15)
        plt.xlabel("Lettre", fontsize=12)
        plt.ylabel("Temps (secondes)", fontsize=12)

        # Ajoute une grille horizontale légère
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Ajuste la mise en page
        plt.tight_layout()

        # Sauvegarde la figure
        plt.savefig(filename, format="PNG", dpi=300)
        print(f"Graphique des temps par niveau sauvegardé dans {filename}")
        plt.close()

    def _create_success_rate_chart(self, filename):
        """Crée un graphique montrant le taux de réussite pour chaque lettre."""
        letters = list(self.stats['success_rate_per_letter'].keys())
        rates = [
            rate * 100 for rate in self.stats['success_rate_per_letter'].values()]

        plt.figure(figsize=(10, 6), facecolor='white')

        # Crée un graphique à barres avec une palette de couleurs dégradée
        bars = plt.bar(letters, rates, color=plt.cm.Blues(
            np.linspace(0.4, 0.8, len(letters))))

        # Ajoute les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1, f"{height:.1f}%",
                     ha='center', va='bottom', fontsize=10)

        # Ajoute un titre et des étiquettes
        plt.title("Taux de réussite par lettre", fontsize=15)
        plt.xlabel("Lettre", fontsize=12)
        plt.ylabel("Taux de réussite (%)", fontsize=12)

        # Définit l'échelle de l'axe y de 0 à 100
        plt.ylim(0, 100)

        # Ajoute une grille horizontale légère
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Ajuste la mise en page
        plt.tight_layout()

        # Sauvegarde la figure
        plt.savefig(filename, format="PNG", dpi=300)
        print(f"Graphique des taux de réussite sauvegardé dans {filename}")
        plt.close()

    def _create_sunburst_chart(self, filename):
        """Crée un diagramme sunburst des chemins."""
        try:
            # Approche alternative avec un treemap
            # Préparation d'une structure d'arbre récursive
            class TreeNode:
                def __init__(self, name, parent=None, level=0):
                    self.name = name
                    self.parent = parent
                    self.level = level
                    self.children = []
                    self.size = 1  # Taille de base

                def add_child(self, child):
                    self.children.append(child)

                def find_child(self, name):
                    for child in self.children:
                        if child.name == name:
                            return child
                    return None

            # Créer l'arbre à partir des chemins
            root = TreeNode("Wikipédia", level=0)

            # Niveau 1: départ (Banane)
            start_title = self.all_paths[0][0][0]
            start_node = TreeNode(start_title, parent=root, level=1)
            root.add_child(start_node)

            # Ajouter tous les chemins à l'arbre
            for path in self.all_paths:
                current_node = start_node
                # Commence à 1 car 0 est le nœud de départ
                for i in range(1, len(path)):
                    title = path[i][0]
                    letter = self.target_word[i-1]
                    node_name = f"{letter}: {title}"

                    # Cherche si ce nœud existe déjà comme enfant
                    child = current_node.find_child(node_name)
                    if child is None:
                        # Crée un nouveau nœud
                        child = TreeNode(
                            node_name, parent=current_node, level=i+1)
                        current_node.add_child(child)
                    else:
                        # Augmente la taille du nœud existant
                        child.size += 1

                    current_node = child

            # Fonction pour convertir l'arbre en listes pour matplotlib
            def tree_to_lists(node, x, y, width, height, dx, dy, rectangles, labels, colors):
                color = plt.cm.tab10(node.level % 10)

                # Ajoute ce nœud
                rectangles.append((x, y, width, height))
                labels.append(node.name)
                colors.append(color)

                # S'il n'y a pas d'enfants, on s'arrête
                if not node.children:
                    return

                # Calcule la taille totale des enfants
                total_size = sum(child.size for child in node.children)

                # Position initiale
                current_x, current_y = x + dx, y + dy
                remaining_width = width - 2*dx
                remaining_height = height - 2*dy

                # Direction du découpage (horizontal ou vertical)
                horizontal = width > height

                # Ajoute les enfants
                for child in node.children:
                    if horizontal:
                        # Divise horizontalement
                        child_width = remaining_width * \
                            (child.size / total_size)
                        tree_to_lists(child, current_x, current_y, child_width, remaining_height,
                                      dx/2, dy/2, rectangles, labels, colors)
                        current_x += child_width
                    else:
                        # Divise verticalement
                        child_height = remaining_height * \
                            (child.size / total_size)
                        tree_to_lists(child, current_x, current_y, remaining_width, child_height,
                                      dx/2, dy/2, rectangles, labels, colors)
                        current_y += child_height

            # Listes pour stocker les informations du treemap
            rectangles = []
            labels = []
            colors = []

            # Crée le treemap
            tree_to_lists(root, 0, 0, 1, 1, 0.01, 0.01,
                          rectangles, labels, colors)

            # Crée la figure
            plt.figure(figsize=(15, 10), facecolor='white')

            # Dessine les rectangles
            for i, (x, y, width, height) in enumerate(rectangles):
                rect = plt.Rectangle((x, y), width, height,
                                     facecolor=colors[i], edgecolor='white', alpha=0.8)
                plt.gca().add_patch(rect)

                # Ajoute les labels si le rectangle est assez grand
                if width * height > 0.01:
                    label = labels[i]
                    if len(label) > 20:
                        label = label[:17] + "..."
                    plt.text(x + width/2, y + height/2, label,
                             ha='center', va='center', fontsize=min(10, width*100),
                             fontweight='bold', color='white' if np.mean(colors[i][:3]) < 0.5 else 'black')

            # Configure les axes
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')

            # Ajoute un titre
            plt.title(
                f"Visualisation arborescente des chemins pour '{self.target_word}'", fontsize=16, pad=20)

            # Crée une légende pour les niveaux
            legend_elements = []
            level_names = ["Wikipédia", "Départ"] + list(self.target_word)
            for i, name in enumerate(level_names):
                if i < len(plt.cm.tab10.colors):
                    legend_elements.append(
                        plt.Rectangle((0, 0), 1, 1, facecolor=plt.cm.tab10(i % 10),
                                      edgecolor='white', label=f"Niveau {name}")
                    )

            plt.legend(handles=legend_elements, loc='upper right',
                       bbox_to_anchor=(1.1, 1.05), fontsize=10)

            # Ajuste la mise en page
            plt.tight_layout()

            # Sauvegarde la figure
            plt.savefig(filename, format="PNG", dpi=300, bbox_inches='tight')
            print(f"Visualisation arborescente sauvegardée dans {filename}")
            plt.close()

        except Exception as e:
            print(
                f"Erreur lors de la création de la visualisation arborescente: {e}")

            # Visualisation alternative plus simple en cas d'erreur
            try:
                plt.figure(figsize=(12, 12), facecolor='white')

                # Crée un simple graphique en secteurs pour montrer la répartition des chemins
                labels = []
                sizes = []
                colors = plt.cm.tab10.colors[:len(self.target_word)]

                # Compte les occurrences de chaque combinaison E->F->R->E->I
                path_counts = Counter(
                    [" → ".join(title for title, _ in path[1:]) for path in self.all_paths])

                # Prend les 10 chemins les plus fréquents
                for path, count in path_counts.most_common(10):
                    labels.append(path)
                    sizes.append(count)

                # Crée le graphique en secteurs
                plt.pie(sizes, labels=None, colors=colors, autopct='%1.1f%%',
                        startangle=90, wedgeprops={'edgecolor': 'white'})

                # Ajoute une légende avec les chemins
                plt.legend(labels, loc='upper right',
                           bbox_to_anchor=(1.2, 1.05), fontsize=8)

                # Ajoute un titre
                plt.title(
                    f"Répartition des chemins pour '{self.target_word}'", fontsize=16)

                # Sauvegarde la figure
                plt.savefig(filename, format="PNG",
                            dpi=300, bbox_inches='tight')
                print(f"Graphique alternatif sauvegardé dans {filename}")
                plt.close()

            except Exception as e2:
                print(f"Échec de la création du graphique alternatif: {e2}")

    def create_html_visualization(self, filename=None):
        """Crée une visualisation HTML interactive des chemins trouvés."""
        if not self.all_paths:
            print("Aucun chemin à visualiser.")
            return

        if filename is None:
            filename = os.path.join(
                self.output_dir, f"{self.target_word}_visualization.html")

        # Prépare les données pour la visualisation
        nodes = []
        links = []
        node_lookup = {}

        # Crée un identifiant unique pour chaque nœud
        for path_idx, path in enumerate(self.all_paths):
            for j, (title, url) in enumerate(path):
                if j == 0:
                    node_id = f"start_{title.replace(' ', '_')}"
                    node_level = "start"
                    node_group = 0
                else:
                    node_id = f"{self.target_word[j-1]}_{path_idx}_{title.replace(' ', '_')}"
                    node_level = self.target_word[j-1]
                    node_group = j

                # Évite les doublons de nœuds
                if node_id not in node_lookup:
                    nodes.append({
                        "id": node_id,
                        "name": title,
                        "level": node_level,
                        "group": node_group,
                        "url": url
                    })
                    node_lookup[node_id] = len(nodes) - 1

                # Ajoute un lien au nœud précédent
                if j > 0:
                    prev_title = path[j-1][0]
                    if j-1 == 0:
                        prev_id = f"start_{prev_title.replace(' ', '_')}"
                    else:
                        prev_id = f"{self.target_word[j-2]}_{path_idx}_{prev_title.replace(' ', '_')}"

                    links.append({
                        "source": node_lookup[prev_id],
                        "target": node_lookup[node_id],
                        "path": path_idx,
                        "value": 1
                    })

        # Prépare les statistiques pour la visualisation
        stats_data = {
            "total_paths": len(self.all_paths),
            "pages_visited": len(self.stats['pages_visited']),
            "total_links": self.stats['total_links_found'],
            "success_rates": {letter: rate * 100 for letter, rate in self.stats['success_rate_per_letter'].items()},
            "time_per_level": self.stats['time_per_level'],
            "total_time": self.stats['total_time']
        }

        # Crée le contenu HTML
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Chemins Wikipedia pour épeler '{self.target_word}'</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .card {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                h1 {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .stats-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                .stat-card {{
                    background-color: #fff;
                    border-radius: 8px;
                    padding: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                    margin: 10px 0;
                }}
                .stat-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                }}
                .charts-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .chart-container {{
                    height: 300px;
                    position: relative;
                }}
                #graph {{
                    width: 100%;
                    height: 600px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    margin-top: 20px;
                    background-color: #f9f9f9;
                }}
                .node {{
                    stroke: #fff;
                    stroke-width: 2px;
                }}
                .link {{
                    stroke-opacity: 0.6;
                }}
                .node text {{
                    pointer-events: none;
                    font-size: 10px;
                    font-weight: bold;
                }}
                .tooltip {{
                    position: absolute;
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    pointer-events: none;
                    font-size: 14px;
                    z-index: 1000;
                }}
                #path-list {{
                    margin-top: 20px;
                    max-height: 400px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 15px;
                }}
                .path-item {{
                    padding: 10px;
                    margin-bottom: 5px;
                    border-bottom: 1px solid #eee;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }}
                .path-item:hover {{
                    background-color: #f0f0f0;
                }}
                .path-item.active {{
                    background-color: #e0f0e0;
                }}
                .controls {{
                    display: flex;
                    justify-content: center;
                    margin: 15px 0;
                    gap: 10px;
                }}
                button {{
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                    transition: background-color 0.2s;
                }}
                button:hover {{
                    background-color: #2980b9;
                }}
                .two-columns {{
                    display: grid;
                    grid-template-columns: 2fr 1fr;
                    gap: 20px;
                }}
                @media (max-width: 768px) {{
                    .two-columns {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Chemins Wikipedia pour épeler '{self.target_word}'</h1>
                
                <div class="card">
                    <h2>Statistiques globales</h2>
                    <div class="stats-container">
                        <div class="stat-card">
                            <div class="stat-label">Chemins trouvés</div>
                            <div class="stat-value">{len(self.all_paths)}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Pages visitées</div>
                            <div class="stat-value">{len(self.stats['pages_visited'])}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Liens trouvés</div>
                            <div class="stat-value">{self.stats['total_links_found']}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Temps total</div>
                            <div class="stat-value">{self.stats['total_time']:.1f}s</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Visualisations</h2>
                    <div class="charts-container">
                        <div class="chart-container">
                            <canvas id="successRateChart"></canvas>
                        </div>
                        <div class="chart-container">
                            <canvas id="timePerLevelChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Réseau des chemins</h2>
                    <div class="controls">
                        <button id="resetZoomBtn">Réinitialiser la vue</button>
                        <button id="centerGraphBtn">Centrer le graphe</button>
                    </div>
                    <div class="two-columns">
                        <div id="graph"></div>
                        <div id="path-list">
                            <h3>Chemins trouvés ({len(self.all_paths)})</h3>
                            <div id="paths"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
            // Données des chemins et des statistiques
            const nodesData = {json.dumps(nodes)};
            const linksData = {json.dumps(links)};
            const pathsData = {json.dumps([
            [title for title, _ in path] for path in self.all_paths
        ])};
            const statsData = {json.dumps(stats_data)};
            
            // Créer les graphiques statistiques
            function createCharts() {{
                // Graphique du taux de réussite
                const successCtx = document.getElementById('successRateChart').getContext('2d');
                const successRateChart = new Chart(successCtx, {{
                    type: 'bar',
                    data: {{
                        labels: Object.keys(statsData.success_rates),
                        datasets: [{{
                            label: 'Taux de réussite (%)',
                            data: Object.values(statsData.success_rates),
                            backgroundColor: [
                                'rgba(54, 162, 235, 0.6)',
                                'rgba(75, 192, 192, 0.6)',
                                'rgba(153, 102, 255, 0.6)',
                                'rgba(255, 159, 64, 0.6)',
                                'rgba(255, 99, 132, 0.6)'
                            ],
                            borderColor: [
                                'rgba(54, 162, 235, 1)',
                                'rgba(75, 192, 192, 1)',
                                'rgba(153, 102, 255, 1)',
                                'rgba(255, 159, 64, 1)',
                                'rgba(255, 99, 132, 1)'
                            ],
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Taux de réussite par lettre (%)',
                                font: {{
                                    size: 16
                                }}
                            }},
                            legend: {{
                                display: false
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 100,
                                title: {{
                                    display: true,
                                    text: 'Taux de réussite (%)'
                                }}
                            }}
                        }}
                    }}
                }});
                
                // Graphique du temps par niveau
                const timeCtx = document.getElementById('timePerLevelChart').getContext('2d');
                const timePerLevelChart = new Chart(timeCtx, {{
                    type: 'line',
                    data: {{
                        labels: Array.from('{self.target_word}'),
                        datasets: [{{
                            label: 'Temps (secondes)',
                            data: statsData.time_per_level,
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 2,
                            tension: 0.1,
                            fill: true,
                            pointBackgroundColor: 'rgba(75, 192, 192, 1)',
                            pointBorderColor: '#fff',
                            pointRadius: 5
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Temps passé par niveau (secondes)',
                                font: {{
                                    size: 16
                                }}
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                title: {{
                                    display: true,
                                    text: 'Temps (secondes)'
                                }}
                            }}
                        }}
                    }}
                }});
            }}
            
            // Configuration du graphe
            function createNetworkGraph() {{
                const width = document.getElementById('graph').clientWidth;
                const height = document.getElementById('graph').clientHeight;
                
                // Crée un svg
                const svg = d3.select("#graph")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
                
                // Crée un groupe pour contenir tous les éléments
                const g = svg.append("g");
                
                // Ajoute une fonction de zoom
                const zoom = d3.zoom()
                    .scaleExtent([0.1, 4])
                    .on("zoom", (event) => {{
                        g.attr("transform", event.transform);
                    }});
                
                svg.call(zoom);
                
                // Crée une simulation de force
                const simulation = d3.forceSimulation(nodesData)
                    .force("link", d3.forceLink(linksData).id(d => d.id).distance(150))
                    .force("charge", d3.forceManyBody().strength(-300))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("x", d3.forceX(width / 2).strength(0.1))
                    .force("y", d3.forceY(height / 2).strength(0.1));
                
                // Crée une échelle de couleurs pour les chemins
                const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
                
                // Crée une info-bulle
                const tooltip = d3.select("body")
                    .append("div")
                    .attr("class", "tooltip")
                    .style("opacity", 0);
                
                // Crée les liens
                const link = g.append("g")
                    .selectAll("line")
                    .data(linksData)
                    .enter()
                    .append("line")
                    .attr("class", "link")
                    .attr("stroke-width", d => Math.sqrt(d.value) * 2)
                    .attr("stroke", d => colorScale(d.path));
                
                // Crée les nœuds
                const nodeGroup = g.append("g")
                    .selectAll(".node-group")
                    .data(nodesData)
                    .enter()
                    .append("g")
                    .attr("class", "node-group")
                    .call(d3.drag()
                        .on("start", dragStarted)
                        .on("drag", dragged)
                        .on("end", dragEnded));
                
                // Ajoute les cercles des nœuds
                nodeGroup.append("circle")
                    .attr("class", "node")
                    .attr("r", d => d.level === "start" ? 12 : 8)
                    .attr("fill", d => d.level === "start" ? "#ff9900" : colorScale(d.group % 10))
                    .on("mouseover", function(event, node) {{
                        tooltip.transition()
                            .duration(200)
                            .style("opacity", 0.9);
                        tooltip.html(`<strong>${node.name}</strong><br/>Niveau: ${node.level}`)
                            .style("left", (event.pageX + 10) + "px")
                            .style("top", (event.pageY - 28) + "px");
                    }})
                    .on("mouseout", function() {{
                        tooltip.transition()
                            .duration(500)
                            .style("opacity", 0);
                    }})
                    .on("click", function(event, d) {{
                        window.open(d.url, '_blank');
                    }});
                
                // Ajoute les étiquettes des nœuds
                nodeGroup.append("text")
                    .attr("dx", 12)
                    .attr("dy", ".35em")
                    .text(d => d.name.length > 15 ? d.name.substring(0, 15) + "..." : d.name);
                
                // Fonctions de gestion des événements de glisser-déposer
                function dragStarted(event, d) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}
                
                function dragged(event, d) {{
                    d.fx = event.x;
                    d.fy = event.y;
                }}
                
                function dragEnded(event, d) {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}
                
                // Anime la simulation
                simulation.on("tick", () => {{
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
                    
                    nodeGroup
                        .attr("transform", d => `translate(${d.x},${d.y})`);
                }});
                
                // Affiche la liste des chemins
                const pathsContainer = d3.select("#paths");
                
                pathsData.forEach((path, i) => {{
                    const pathElem = pathsContainer.append("div")
                        .attr("class", "path-item")
                        .attr("data-path", i)
                        .html(`<strong>Chemin ${i+1}:</strong> ${path.join(" → ")}`)
                        .on("mouseover", function() {{
                            highlightPath(i);
                        }})
                        .on("mouseout", function() {{
                            resetHighlight();
                        }})
                        .on("click", function() {{
                            d3.selectAll(".path-item").classed("active", false);
                            d3.select(this).classed("active", true);
                            fixHighlight(i);
                        }});
                }});
                
                let highlightFixed = false;
                let fixedPathIndex = -1;
                
                function highlightPath(pathIndex) {{
                    if (highlightFixed && pathIndex !== fixedPathIndex) return;
                    
                    // Atténue tous les liens et nœuds
                    link.style("opacity", 0.1);
                    nodeGroup.style("opacity", 0.1);
                    
                    // Met en évidence les liens du chemin sélectionné
                    link.filter(d => d.path === pathIndex)
                        .style("opacity", 1)
                        .style("stroke-width", 3);
                    
                    // Met en évidence les nœuds du chemin sélectionné
                    const nodesInPath = new Set(link.filter(d => d.path === pathIndex).data().flatMap(d => [d.source.id, d.target.id]));
                    nodeGroup.filter(d => nodesInPath.has(d.id))
                        .style("opacity", 1);
                }}
                
                function resetHighlight() {{
                    if (highlightFixed) {{
                        highlightPath(fixedPathIndex);
                        return;
                    }}
                    
                    link.style("opacity", 0.6)
                        .style("stroke-width", d => Math.sqrt(d.value) * 2);
                    nodeGroup.style("opacity", 1);
                }}
                
                function fixHighlight(pathIndex) {{
                    highlightFixed = true;
                    fixedPathIndex = pathIndex;
                    highlightPath(pathIndex);
                }}
                
                // Bouton pour réinitialiser la vue
                d3.select("#resetZoomBtn").on("click", function() {{
                    highlightFixed = false;
                    fixedPathIndex = -1;
                    d3.selectAll(".path-item").classed("active", false);
                    resetHighlight();
                }});
                
                // Bouton pour centrer le graphe
                d3.select("#centerGraphBtn").on("click", function() {{
                    svg.transition().duration(750).call(
                        zoom.transform,
                        d3.zoomIdentity.translate(width/2, height/2).scale(1)
                    );
                }});
                
                return simulation;
            }}
            
            // Initialise les visualisations
            document.addEventListener('DOMContentLoaded', function() {{
                createCharts();
                createNetworkGraph();
            }});
            </script>
        </body>
        </html>
        '''

        # Enregistre le fichier HTML
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Visualisation HTML interactive sauvegardée dans {filename}")
        print(f"Ouvrez ce fichier dans un navigateur pour explorer les chemins de manière interactive.")


def main():
    # Parse les arguments
    parser = argparse.ArgumentParser(
        description='Trouve des chemins sur Wikipédia qui épellent un mot.')
    parser.add_argument('--start', default='https://fr.wikipedia.org/wiki/Banane',
                        help='URL de départ sur Wikipédia (par défaut: Banane)')
    parser.add_argument('--word', default='EFREI',
                        help='Mot à épeler avec les premières lettres des titres (par défaut: EFREI)')
    parser.add_argument('--max-paths', type=int, default=10,
                        help='Nombre maximum de chemins à trouver (par défaut: 10)')
    parser.add_argument('--links-per-level', type=int, default=3,
                        help='Nombre de liens à explorer à chaque niveau (par défaut: 3)')
    parser.add_argument('--variants-i', type=int, default=5,
                        help='Nombre de variantes spécifiques pour la lettre I (par défaut: 5)')
    parser.add_argument('--output-dir', default='results',
                        help='Dossier pour enregistrer les résultats (par défaut: results)')
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Désactive la création des visualisations (plus rapide)')
    parser.add_argument('--cache-file', default='wiki_cache.json',
                        help='Fichier de cache pour les requêtes Wikipédia (par défaut: wiki_cache.json)')

    args = parser.parse_args()

    # Configuration personnalisée pour chaque lettre
    # Par défaut, utilise le nombre de liens par niveau spécifié
    variants_per_letter = {
        letter: args.links_per_level for letter in args.word}

    # Si le mot contient 'I', utilise le nombre spécifique de variantes pour 'I'
    if 'I' in args.word:
        variants_per_letter['I'] = args.variants_i

    # Affiche la configuration
    print(f"Configuration:")
    print(f"  Mot à épeler: {args.word}")
    print(f"  Nombre de liens par niveau: {args.links_per_level}")
    print(f"  Variantes spécifiques par lettre:")
    for letter, count in variants_per_letter.items():
        print(f"    {letter}: {count}")
    print(f"  Fichier de cache: {args.cache_file}")
    print(f"  Dossier de sortie: {args.output_dir}")
    print("")

    try:
        # Crée l'instance du chercheur de chemins
        finder = WikipathFinder(
            start_url=args.start,
            target_word=args.word,
            max_paths=args.max_paths,
            variants_per_letter=variants_per_letter,
            links_per_level=args.links_per_level,
            cache_file=args.cache_file,
            output_dir=args.output_dir
        )

        # Lance la recherche
        print(
            f"Recherche de chemins pour épeler '{args.word}' à partir de {args.start}")
        paths = finder.find_paths()

        # Affiche et enregistre les résultats
        if not paths:
            print("Aucun chemin complet trouvé.")
        else:
            print(f"Trouvé {len(paths)} chemins complets:")
            finder.print_paths()
            finder.save_paths_to_file()

            # Crée les visualisations si demandé
            if not args.no_visualizations:
                finder.create_visualizations()
            else:
                print("Création des visualisations désactivée.")

    except KeyboardInterrupt:
        print("\nRecherche interrompue par l'utilisateur.")
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")


if __name__ == "__main__":
    main()
