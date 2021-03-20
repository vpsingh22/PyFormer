import re


class LexicalAnalyzer:
    # Token row
    lin_num = 1

    def tokenize(self, code):
        rules = [
            ('LBRACKET', r'\('), 
            ('RBRACKET', r'\)'),
            ('LSQUARE', r'\['), 
            ('RSQUARE', r'\]'), 
            ('LBRACE', r'\{'),          
            ('RBRACE', r'\}'), 
            ('COMMA', r','),   
            ('SCOLON', r';'),
            ('DCOLON', r'::'),
            ('COLON', r':'), 
            ('DOT', r'\.'),
            ('EQ', r'=='), 
            ('NE', r'!='), 
            ('LE', r'<='), 
            ('GE', r'>='), 
            ('TILDE', r'~'),
            ('XOREQ', r'\^='),
            ('XOR', r'\^'),
            ('OR', r'\|\|'), 
            ('BITOR', r'\|'), 
            ('AND', r'&&'), 
            ('BITAND', r'&'),
            ('ATTR', r'\='),
            ('LSHIFT', r'<<'),
            ('RSHIFT', r'>>'),
            ('LT', r'<'),
            ('GT', r'>'),
            ('ADDASN', r'\+='),
            ('PLUS', r'\+'),
            ('SUBASN', r'\-='),
            ('MINUS', r'-'),
            ('POWER', r'\*\*'), 
            ('MULASN', r'\*='),
            ('MULT', r'\*'), 
            ('INTDIVASN', r'\/\/='),
            ('INTDIV', r'\/\/'),
            ('DIVASN', r'\/='),
            ('DIV', r'\/'), 
            ('MODULO', r'%'), 
            ('ID', r'[_a-zA-Z]\w*'), 
            ('FLOAT_CONST', r'\d(\d)*\.\d(\d)*'),
            ('INTEGER_CONST', r'\d(\d)*'), 
            ('NEWLINE', r'\n'),  
            ('TAB', r'[\t]'),
            ('SPACE', r' '),
            ('ESCAPE_SEQ', r'\\.'),
            ('LINEBREAK', r'\\'),
#             ('APOSTROPHE', r'\''),
            ('COMMENT', r'#.*'),
            ('S_MULTILINECOMMENT', r"'''[^'\\\\]*(?:(?:\\\\.|'{1,2}(?!'))[^'\\\\]*)*'''"),
            ('D_MULTILINECOMMENT', r'"""[^"\\\\]*(?:(?:\\\\.|"{1,2}(?!"))[^"\\\\]*)*"""'),
            ('S_STRING_CONST', r'\'.*?\'[^\']'),
            ('D_STRING_CONST', r'".*?"[^"]'),
            ('MISMATCH', r'.'), 
            
            
            
        ]

        tokens_join = '|'.join('(?P<%s>%s)' % x for x in rules)
        lin_start = 0

        # Lists of output for the program
        token = []
        lexeme = []
        row = []
        column = []

        # It analyzes the code to find the lexemes and their respective Tokens
        for m in re.finditer(tokens_join, code):
            token_type = m.lastgroup
            token_lexeme = m.group(token_type)


            if token_type == 'MISMATCH':
                raise RuntimeError('%r unexpected on line %d' % (token_lexeme, self.lin_num))
            else:
                    col = m.start() - lin_start
                    column.append(col)
                    token.append(token_type)
                    lexeme.append(token_lexeme)
                    row.append(self.lin_num)
                    # To print information about a Token
                    # print('Token = {0}, Lexeme = \'{1}\', Row = {2}, Column = {3}'.format(token_type, token_lexeme, self.lin_num, col))

        return token, lexeme, row, column