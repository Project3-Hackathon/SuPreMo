import sys
import subprocess
import os

def filter_snps_vcf(input_vcf, filtered_vcf_file):
    # Perform SNP filtering using bcftools query
    print("input VCF is filtered to retain SNPs only")
    subprocess.run(
        f'bcftools query -f "%CHROM\\t%POS\\t%ID\\t%REF\\t%ALT\\t%QUAL\\t%FILTER\\t%INFO\\t%FORMAT\\n" -i \'TYPE="snp"\' {input_vcf} > {filtered_vcf_file}',
        shell=True,
        check=True
    )

def attach_header(input_vcf, filtered_vcf_file, vcf_header, filtered_vcf_w_header_file, filtered_vcf_w_header_file_zipped):
    # Preserve the initial header, reattach to the filtered VCF, zip and index
    subprocess.run(f'bcftools view -h {input_vcf} > {vcf_header}', shell=True, check=True)
    subprocess.run(f'cat {vcf_header} {filtered_vcf_file} > {filtered_vcf_w_header_file}', shell=True, check=True)
    subprocess.run(f'bgzip {filtered_vcf_w_header_file}', shell=True, check=True)
    subprocess.run(f'bcftools index {filtered_vcf_w_header_file_zipped}', shell=True, check=True)
    # Clean up intermediate files
    subprocess.run(f'rm {vcf_header}', shell=True, check=True)
    subprocess.run(f'rm {filtered_vcf_file}', shell=True, check=True)

def remove_file_extension(file_path, extensions):
    file_basename = os.path.basename(file_path)
    
    for extension in extensions:
        if file_basename.endswith(extension):
            file_basename = file_basename[:-len(extension)]
            break
    
    return file_basename

extensions_to_remove = [".vcf", ".vcf.gz"]

def generate_consensus_fasta(filtered_vcf_w_header_file_zipped, input_fasta, output_fasta_file):
    # Generate consensus fasta using bcftools consensus
    print("personalized genome is created")
    subprocess.run(f'cat {input_fasta} | bcftools consensus {filtered_vcf_w_header_file_zipped} > {output_fasta_file}', shell=True, check=True, stderr=subprocess.DEVNULL)
    
    # Clean up intermediate files
    subprocess.run(f'rm {filtered_vcf_w_header_file_zipped}', shell=True, check=True)
    subprocess.run(f'rm {filtered_vcf_w_header_file_zipped + ".csi"}', shell=True, check=True)

def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Please provide two input file paths as arguments. Arg1 = personalized SNP vcf, Arg2 = Reference genome fasta")
        sys.exit(1)

    input_vcf = sys.argv[1]
    input_fasta = sys.argv[2] # this should be the reference genome 

    # Create a subdirectory called "personalized_genome" in the "data" directory to store the temp files as well as the personalized genome fasta 
    data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
    temp_directory = os.path.join(data_directory, 'personalized_genome')
    os.makedirs(temp_directory, exist_ok=True)

    # Define paths to temporary and output files
    input_vcf_without_extension = remove_file_extension(input_vcf, extensions_to_remove)
    filtered_vcf_file = os.path.join(temp_directory, input_vcf_without_extension + '.filtered.vcf')
    vcf_header = os.path.join(temp_directory, input_vcf_without_extension + '.vcf.header')
    filtered_vcf_w_header_file = os.path.join(temp_directory, input_vcf_without_extension + '.filtered_w_header.vcf')
    filtered_vcf_w_header_file_zipped = filtered_vcf_w_header_file + '.gz'
    output_fasta_file = os.path.join(temp_directory, input_vcf_without_extension + '.consensus.fa')


    # Step 1: VCF preprocessing: Filter SNPs from input VCF, prepare for generation of personalized ref genome
    # Step 1.a: Filter SNPs from VCF
    filter_snps_vcf(input_vcf, filtered_vcf_file)

    # Step 1.b: Re-attach header, zip and index the filtered VCF
    attach_header(input_vcf, filtered_vcf_file, vcf_header, filtered_vcf_w_header_file, filtered_vcf_w_header_file_zipped)

    # Step 2: Generate consensus fasta
    generate_consensus_fasta(filtered_vcf_w_header_file_zipped, input_fasta, output_fasta_file)
    
if __name__ == "__main__":
    main()


