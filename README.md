# Niching Methods in Evolutionary Algorithms

**Languages & Tools:** Python, DEAP, NumPy, Matplotlib, Jupyter Notebook

---

## **Project Overview**

This project demonstrates and compares several niching methods in evolutionary algorithms to maintain population diversity and discover multiple optima in complex multimodal landscapes.

Implemented in Python using the DEAP evolutionary computation library, it includes:

- *Crowding*
- *Fitness Sharing*
- *Clearing*
- *Speciation*

Each method is applied to a predefined multimodal fitness function, with results visualized across generations.

---

## **Key Features**

- Visual analysis of population distribution and convergence  
- Modular design for extensibility and experimentation  
- Supports both *maximization* and *minimization* objectives  
- Customizable evolutionary parameters and niching strategies  
- DEAP-powered genetic algorithm implementation  

---

## **Niching Methods Implemented**

| **Method**          | **Description**                                                                 |
|---------------------|---------------------------------------------------------------------------------|
| *Crowding*          | Preserves diversity by replacing similar individuals with fitter offspring      |
| *Fitness Sharing*   | Penalizes fitness in dense areas to encourage niche formation                    |
| *Clearing*          | Only a limited number of top individuals survive in each niche                  |
| *Speciation*        | Groups individuals into species based on similarity and evolves them separately |

---

## **How to Run**

**1. Clone the repository**

```bash
git clone https://github.com/sprobulski/evolutionary-niching-algorithms.git
cd evolutionary-niching-algorithms
```
**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Launch the notebook**

```bash
jupyter notebook evolutionary-niching-algorithms.ipynb
```

## **Skills Demonstrated**

- Evolutionary Computation & Niching Strategies
- Scientific Visualization
- Python OOP and modular code structure
- Use of DEAP (Distributed Evolutionary Algorithms in Python)
