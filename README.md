README.md

[Hungry Geese](https://www.kaggle.com/c/hungry-geese/overview) was a competitive simulation challenge run by Kaggle in 2021. It is a 4 player survival game where teams create agents (‘geese’) to compete against other teams' agents. It resembled the evergreen snake game, where your snake or goose had to stay alive on a small board amongst others.

This is the solution from our team "THE GOOSE IS LOOSE". In a field of 879 teams, we were placed 7th on the day of submission close. A two week play off period will reveal our true position somewhere in the top 20 but we hope for a top 11 gold medal. You can read our lengthy [solution description](https://kaggle.com). 

Our solution is a single file agent as submitted to kaggle and includes

- A Reinforcement Learning NN trained with a customised [HandyRL](https://github.com/DeNA/HandyRL) framework. 
    
- A lookahead (LA) setup similar in nature to the [Stockfish](https://www.chessprogramming.org/Stockfish) chess engine
    
- A [Floodfill](https://en.wikipedia.org/wiki/Flood_fill) algorithm to determine influence areas and desirability of moves.
    
- Bespoke heuristics / rules as guard rails and meta logic.
   



Our team was

- Rob Gardiner https://www.kaggle.com/robga https://github.com/digitalspecialists
    
- Taaha Khan https://www.kaggle.com/taahakhan https://github.com/taaha-khan
    
- Anton https://www.kaggle.com/superant https://github.com/Gerenuk
    
- Corey Levison https://www.kaggle.com/returnofsputnik https://github.com/Quetzalcohuatl
    
- Kha Vo https://www.kaggle.com/khahuras https://github.com/voanhkha