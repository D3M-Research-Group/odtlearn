cd ~/project-template/
pyreverse odtlearn  --colorized --ignore utils tests -a 2 -d ~/project-template/figures/ -f 'SPECIAL'

sed -i '' -e 's/color="aliceblue"/fillcolor="aliceblue"/g' ~/project-template/figures/classes.dot
sed -i '' -e 's/label=\"{ABC.*\}/label="\{Abstract Base Class\}/g' ~/project-template/figures/classes.dot
sed -i '' -e 's/, style="filled"];/, style="filled", fontsize="25"];/g' ~/project-template/figures/classes.dot
sed -i '' -e 's/charset="utf-8"/charset="utf-8";ranksep=0.2;/g' ~/project-template/figures/classes.dot
sed -i '' -e 's/plot\_tree\(.*\)\\lp/plot\_tree\(\.\.\.\)\\lp/g' ~/project-template/figures/classes.dot
sed -i '' -e 's/[ ]:[ ][a-zA-Z]*\\l/\\l/g' ~/project-template/figures/classes.dot
sed -i '' -e 's/[ ]:[ ]\_[a-zA-Z]*\\l/\\l/g' ~/project-template/figures/classes.dot
sed -i '' -e 's/[ ]:[ ][A-Z][a-z]*[A-Z][a-z]*,[ ][A-Z][a-z]*[A-Z][a-z]*\\l/\\l/g' ~/project-template/figures/classes.dot

rm -f ~/project-template/figures/classes.dot-e
rm -f ~/project-template/figures/packages.dot

# dot -Tpng ~/project-template/figures/classes.dot -o ~/project-template/figures/odtlearn_uml.png
python figures/edit_dot.py "figures/classes.dot" "figures/odtlearn_uml.png"