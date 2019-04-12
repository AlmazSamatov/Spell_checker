# Spell checker

## Project description. 
### Main idea behind solution. 
  Projects main idea is spell-checker which takes into account keyboard layout i.e. when it calculates edit distance it gives less error points to letters which are near in QWERTY keyboard.  It is logically clear that mistypings on keyboard is main source of misspellings. I think reason in widespread use of smartphones for different tasks such as browsing, messaging etc., because typing correctly on smartphone is a hard task. This is the reason why I decided to investigate this problem.  In the process different solutions was tested and most effective was chosen.  

## Solution explain.
  First of all, I want to mention that I took Damerau–Levenshtein distance as edit distance metric, because it takes into consideration transpositions (which is really important in misspellings) compared by default Levenshtein distance. Then I constructed QWERTY keyboard graph. Function for finding shortest path between two letters uses Breadth First Search algorithm.  Damerau–Levenshtein distance: 
 
  In usual Damerau–Levenshtein distance, there are all error points are equal to 1, but in our case they need to be weighted. I tried different solutions for calculating edit distance:  
  1) As in thesis (Thesis “Weighting Edit Distance to Improve Spelling Correction in Music Entity Search”: http://www.nada.kth.se/~ann/exjobb/axel_samuelsson.pdf), mapping substitution error points to range from 0 to 2 and 0.7 to 1.7 (If they are close then error point will be less than 1. Deletion error points is weighted by the average of the distances to the adjacent characters in the string. And 1 for transpositions and insertion. (Gives 60% correctness on test) 
  2) Considering only neighbour letters on distance 1. For example, give 0.5 error point if letters are neighbour, else give 1. Deletion error points is weighted by the average of the distances to the adjacent characters in the string. And 1 for transpositions and insertion. (Gives 59% correctness on test) 
  3) Considering only neighbour letters on distance 2, if further then return 3, else return distance itself (1 or 2). Deletion error points are equal to 2; insertion depends: if we insert the same letter as before then give 1 (for example, in given word: “localy” and correction “locally”, inserting second ‘l’ will have weight 1) else 3. And give 1 for transposition. (Gives 69% correctness on test) 
  
  In my opinion, first solution is not so good, because it takes into account all distances between letters while only neighbouring letters are important and second one is too restrict, because it is interested only in letters in distance 2. I came up with 3rd solution, because it is some kind of compromise. It is clear that transposition error points should be less than deletion, insertion (if not insertion of the same letter), substitution of the letters, which are far. 
And same explanation are for other error points (deletion is some kind of mid range task, then give to it 2 points; insertion of the same letter is before is 1 point, because this is little mistake (misspelling)).  
  
  In the process I also found several different solutions to improve spell checker: first of them is that if no known edited word found in text corpus then I stem word and try to correct root and then return it in the following format: corrected root + suffix (it assumes that error in root but suffix of word is correct), second is dropping edited words, which begins by different letter than original word, because in most cases people never make errors in the beginning of words.  
 

## Results.
| Original word | Corrected by keyboard layout mode | Corrected without keyboard layout mode  |
| ------------- |:-------------:| -----:|
|  coetect      | correct | contact |
| cobhect      | connect      |   correct |
| jucie | juice      |    julie |
| localy | locally      |    local |
| geniva | geneva      |    genius |
| futher | further   |  father  | 
