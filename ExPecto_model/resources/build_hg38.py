#!/usr/bin/env python
# coding: utf-8


import os,sys,argparse

def ParseArg():
    ''' This Function Parse the Argument '''
    p=argparse.ArgumentParser( description = 'Example: %(prog)s -h', epilog='Library dependency :')
    p.add_argument('-v','--version',action='version',version='%(prog)s 0.1')
    p.add_argument('-g','--gtf',type=str,dest="gtf",help="input gtf file")
    p.add_argument('--gene_list',type=str,dest="gene_list",help="gene_list file")
    p.add_argument('--output',type=str,dest="output",help="output file")
    if len(sys.argv) < 2:
        print(p.print_help())
        exit(1)
    return p.parse_args()


def main():
    global args
    args=ParseArg()
    # load gene list
    gene_table = {}
    with open(args.gene_list, 'r') as fin:
        for line in fin:
            if line.strip() == '':
                continue
            row = line.strip().split(',')
            if row[0] == 'id':
                continue
            gene_id = row[0]
            gene_table[gene_id] = row[0]
    # parse gtf file 
    gene_list = []
    with open(args.gtf, 'r') as fin:
        for line in fin:
            if line.strip().startswith('#'):
                continue
            row = line.strip().split('\t')
            # chr1    HAVANA  gene    11869   14409   .       +       .       gene_id "ENSG00000290825.1"; gene_type "lncRNA"; gene_name "DDX11L2"; level 2; tag "overlaps_pseudogene";
            if row[2] != 'gene':
                continue
            chrom = row[0]
            source = row[1]
            feature = row[2]
            start = int(row[3])
            end = int(row[4])
            score = row[5]
            strand = row[6].strip()
            phase = row[7]
            annotation = row[8].strip().split(';')
            # 
            attr = {}
            for item in annotation:
                if item.strip() == '':
                    continue
                record = item.strip().split()
                key = record[0].strip()
                value = record[1].strip().replace('"', '')
                attr[key] = value
            if attr.get('gene_id', None) is not None:
                gene_id = attr['gene_id'].split('.')[0]
                if gene_table.get(gene_id, None) is not None:
                    gene = {'gene_id': gene_id,
                            'gene_name': attr['gene_name'],
                            'gene_type': attr['gene_type'],
                            'chrom': chrom,
                            'start': start - 1,
                            'end': end,
                            'strand': strand
                            }
                    gene_list.append(gene)
    #
    with open(args.output, 'w') as fout:
        for gene in gene_list:
            if gene['strand'] == '+':
                tss_start = gene['start']
                tss_end =tss_start + 1
            elif gene['strand'] == '-':
                tss_start =  gene['end'] - 1
                tss_end = tss_start + 1
            print("%s\t%d\t%d\t%s\t%s" % (gene['chrom'], tss_start, tss_end, gene['strand'], gene['gene_id']), file = fout)


if __name__=="__main__":
    main()

