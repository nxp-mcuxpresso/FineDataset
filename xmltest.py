import xml.dom.minidom

def ParseNode(node):
    if node.firstChild == node.lastChild:
        return node.firstChild.nodeValue
    else:
        dct2 = dict()
        for child in node.childNodes:
            keyName = child.localName
            if keyName is None:
                continue
            dct2[keyName] = ParseNode(child)
        return dct2

dom1=xml.dom.minidom.parse('q:/tmp/2008_000002.xml')
root=dom1.documentElement
dct = dict()
for node in root.childNodes:
    if node.nodeType == 3 and node.firstChild is None:
        # 字符串节点
        if node.localName is None:
            # 缩进节点
            continue
    elif node.nodeType == 1:
        # element节点
        dct[node.localName] = ParseNode(node)
print(root.childNodes)

