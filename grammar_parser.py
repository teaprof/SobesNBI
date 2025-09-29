from problem_ext import ExtendedProblem
import lark
import numpy as np

class DataExtractor(lark.Transformer):
    """Extract data from AST created by lark parser"""
    def __init__(self) : 
        lark.Transformer.__init__(self)
        self.d = dict()
        self.t = dict()
        self.p = dict()
        self.f = dict()        
        self.before_list = []
        self.product_names = []
        
    def before(self, tree):
        # the name of this method must correspond to the name of grammar rule
        name1 = tree[0].value
        name2 = tree[1].value
        self.before_list.append((name1, name2))
        
        
    def product(self, tree):
        # the name of this method must correspond to the name of grammar rule
        name = tree[0].value
        self.t[name] = float(tree[1].value)
        self.d[name] = float(tree[2].value)
        self.p[name] = float(tree[3].value)
        self.product_names.append(name)
        
    
    def fine(self, tree):
        # the name of this method must correspond to the name of grammar rule
        name = tree[0].value
        fine = float(tree[1].value)
        self.f[name] = fine
        
class GrammarParser:
    grammar = \
        """
        start: products befores fines
        products: product *
        product: "product" NAME "=" FLOAT "," FLOAT  "," FLOAT
        befores: before *
        before: NAME "before" NAME
        fines: fine *
        fine: "fine for" NAME "is" FLOAT

        FLOAT: /[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?/        
        NAME: /[a-zA-Z][a-zA-Z0-9]*/
        WHITESPACE: (" " | "\\n")+
        COMMENT: "#" /[^\\n]/*
        %ignore WHITESPACE
        %ignore COMMENT
        """
    def __init__(self):
        pass
    
    def parse(self, description) -> ExtendedProblem:
        # Parse strings and build AST
        parser = lark.Lark(GrammarParser.grammar)
        tree = parser.parse(description)
        #print(tree)
        
        # Extract data from AST
        extractor = DataExtractor()
        extractor.transform(tree)
        
        # Initialize ExtendedProblem object
        nProducts = len(extractor.d)        
        res = ExtendedProblem(nProducts, np.zeros(nProducts), np.zeros(nProducts), np.zeros(nProducts))
        for name, t in extractor.t.items():
            n = extractor.product_names.index(name)
            res.t[n] = np.asarray(t)
        for name, d in extractor.d.items():
            n = extractor.product_names.index(name)
            res.d[n] = np.asarray(d)
        for name, p in extractor.p.items():
            n = extractor.product_names.index(name)
            res.p[n] = np.asarray(p)
        for name1, name2 in extractor.before_list:
            n1 = extractor.product_names.index(name1)
            n2 = extractor.product_names.index(name2)
            res.befores.append((n1, n2))
        res.fines = np.zeros(nProducts)
        for name, fine in extractor.f.items():
            n = extractor.product_names.index(name)
            res.fines[n] = fine
        return res                
        
        
