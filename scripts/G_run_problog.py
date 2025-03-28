import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problog.program import PrologString
from problog.formula import LogicFormula, LogicDAG
from problog.ddnnf_formula import DDNNF
from problog import get_evaluatable

def run_problog_analysis():
    # Read the Prolog file
    with open('automated_reasoning.pl', 'r') as f:
        program = f.read()
    
    # Create a Prolog program
    p = PrologString(program)
    
    # Ground the program
    lf = LogicFormula.create_from(p)
    
    # Convert to CNF
    dag = LogicDAG.create_from(lf)
    
    # Convert to d-DNNF
    ddnnf = DDNNF.create_from(dag)
    
    # Evaluate and get probabilities
    results = ddnnf.evaluate()
    
    # Print results
    print("\nProblog Analysis Results:")
    print("------------------------")
    for query, probability in results.items():
        print(f"Query: {query}")
        print(f"Success Probability: {probability}")

if __name__ == "__main__":
    run_problog_analysis() 