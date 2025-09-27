import numpy as np
import dataclasses
import csv

@dataclasses.dataclass
class Problem:
    # number of pruducts
    nProducts: int = 9
    # processing durations for each product
    t: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    # deadlines when each product should be finished
    d: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    # rewards for each product got when it is finished before deadline
    p: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
        
    # generate random problem
    def randomize(self, nProduct = None):        
        if nProduct is None:
            self.nProducts = np.random.randint(1,10)
        else:
            self.nProducts = nProduct
        self.t = np.random.rand(self.nProducts)
        self.d = np.random.rand(self.nProducts)*self.nProducts/2
        #self.p = np.ones(self.nProducts) 
        self.p = np.random.rand(self.nProducts)            
    
    # pretty printing
    def __str__(self):
        str1 = f"nProducts = {self.nProducts}\n"
        str2 = "t = " + str(self.t)
        str3 = "d = " + str(self.d)
        str4 = "p = " + str(self.p)
        return str1 + "\n" + str2 + "\n" + str3 + "\n" + str4
    
    # calcualte the reward for the specified schedule
    def objectiveFcn(self, schedule: list[int]) -> float:
        assert len(schedule) == self.nProducts
        assert len(set(schedule)) == self.nProducts
        tend = np.cumsum(self.t[schedule])
        bonus = 0
        for n in range(self.nProducts):
            if tend[n] < self.d[schedule[n]]:
                bonus += self.p[schedule[n]]
        return bonus
        # The following code is slower than above one approx 2 times:
        #bonus = np.sum(self.p[schedule]*(tend - self.d[schedule] <= 0))
        #return bonus                

    # check if the solution is valid    
    def checkSolution(self, schedule: list[int]) -> bool:
        if len(schedule) != self.nProducts:
            return False
        if len(set(schedule)) != self.nProducts:
            return False
        return True
    
    def saveCSV(self, filename):
        res = ""
        for n in range(self.nProducts):
            res += f"{self.t[n]:10}, {self.d[n]:10}, {self.p[n]:10}\n"
        with open(filename, "w") as f:
            f.write(res)
    
    def readCSV(self, filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            d, t, p = [], [], []        
            for row in reader:
                t.append(float(row[0]))
                d.append(float(row[1]))
                p.append(float(row[2]))
        self.nProducts = len(t)
        self.t = np.asarray(t)
        self.d = np.asarray(d)
        self.p = np.asarray(p)