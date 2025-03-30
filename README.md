# debunkolicard
Un projet qui met à mal ce fourbe et malicieux Fabien Olicard.

![olicard](https://github.com/user-attachments/assets/a3183347-389f-444e-a706-11c972eeb430)

## 🧐 L'Affaire Olicard

Lors d'un spectacle à l'EFREI, Fabien Olicard a réalisé un tour de "mentalisme" qui nous a laissés perplexes. Voici ce qui s'est passé :

1. Il fait monter une étudiante sur scène
2. Il lui demande d'ouvrir Wikipédia sur n'importe quelle page (en l'occurrence "Banane")
3. Il la laisse naviguer librement sur les liens
4. À chaque nouvelle page, il note le titre sur son paperboard
5. À la fin, MIRACLE ! Les lettres des pages forment "EFREI" 🤯

![fabien](https://github.com/user-attachments/assets/8e3ba120-dfe3-4c70-9bb5-b03be4d67f13)

**MAIS ATTENDEZ...** 🚨 Quelque chose cloche ici. Nous avons essayé de reproduire le même parcours que lors du spectacle et... aucune mention de "théâtre" ou de "theatre" sur la page _chauffage_.

![2025-03-30 18 09 23](https://github.com/user-attachments/assets/ab8c1d29-c546-4669-96df-48d60620d696)

**IMPOSSIBLE** de trouver les mêmes chemins ! Nous avons alors créé ce projet pour démasquer la supercherie.

## 🔍 Nos découvertes suspectes

1. Olicard n'a pas choisi les **premières lettres** des pages, mais plutôt **les lettres qui l'arrangeaient** 🧙‍♂️✨
2. En parcourant systématiquement les liens, nous n'avons pas pu reproduire le chemin exact montré pendant le spectacle 🤔
3. Notre analyse statistique montre que certaines lettres sont pratiquement impossibles à trouver dans un parcours aléatoire 📊

![EFREI_network](https://github.com/user-attachments/assets/d98a07b3-053f-43a5-bda3-5e8bfa6a067b)

## 💻 Notre outil de débunkage

Nous avons créé un algorithme qui tente de trouver TOUS les chemins possibles pour épeler "EFREI" (ou n'importe quel mot) en navigant dans Wikipédia. Les résultats sont sans appel :

![EFREI_time_per_level](https://github.com/user-attachments/assets/dd4525cb-53c0-493d-a89c-02c1f3773b7f)

![EFREI_top_titles_I](https://github.com/user-attachments/assets/163e7b43-0def-42f8-a104-ab85d3c21549)

## 🛠️ Installation et utilisation

Pour mener votre propre enquête :

```bash
# Clonez le repo
git clone https://github.com/votre_username/debunkolicard.git

# Installez les dépendances
pip install -r requirements.txt

# Lancez l'outil de débunkage
python debunk.py
```

Options disponibles :
```
python debunk.py --start https://fr.wikipedia.org/wiki/Banane --word EFREI --max-paths 10
```

## 🔮 Conclusion

Cet outil prouve qu'il est pratiquement impossible de réaliser ce tour par "hasard" ou "mentalisme". Plusieurs hypothèses demeurent :

1. Fabien avait préparé le tour à l'avance avec la participante 🤝
2. Il a modifié discrètement les pages/titres qu'il écrivait 📝
3. Il a utilisé la technique classique du "forcing" mental 🧠
4. C'est véritablement un sorcier 🧙‍♂️ (hypothèse peu probable mais non exclue)
