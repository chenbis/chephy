library(immunarch)
library(dplyr)

downsample_clonotypes <- function(num_clonotypes = 10000, threshold = 10000, prob = TRUE) {

  folders <- list(
    healthy = c("/dsi/sbm/chen2/raw_data/healthy/TRB_h_v1", "/dsi/sbm/chen2/raw_data/healthy/TRB_h_v2"),  # Adjust paths
    sick = "/dsi/sbm/chen2/raw_data/sick/mixcr_4.7_TRB"
  )

  file_extension <- "\\.tsv$"              

  output_dir <- paste0("/dsi/sbm/chen2/downsampled_data/", num_clonotypes)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  convert_to_immunarch <- function(df) {
    df %>%
      select(Clones = readCount, 
             CDR3.aa = aaSeqCDR3,
             V.name = allVHitsWithScore,
             J.name = allJHitsWithScore) %>%
      mutate(V.name = sub("\\*.*$", "", V.name),
             J.name = sub("\\*.*$", "", J.name)) %>%
      filter(CDR3.aa != "") %>%
      group_by(CDR3.aa, V.name, J.name) %>%
      summarise(Clones = sum(Clones), .groups = 'drop') %>%
      arrange(desc(Clones))
  }

  for (type in names(folders)) {
    
    paths <- folders[[type]]
    
    for (path in paths) {
      
      files <- list.files(path = path, pattern = file_extension, full.names = TRUE)
      
      for (file in files) {
        
        cat("Processing file:", file, "\n")
        
        df_original <- read.delim(file, header = TRUE, stringsAsFactors = FALSE)
        original_filename <- basename(file)
        original_ext <- tools::file_ext(file)
        
        df_immunarch <- convert_to_immunarch(df_original)
        
        if (nrow(df_immunarch) < threshold) {
          cat(" - Skipped (below threshold:", nrow(df_immunarch), "clonotypes)\n")
          next
        }
        
        df_downsampled <- repSample(df_immunarch, .method = "sample", .n = num_clonotypes, .prob = prob)
        
        final_df <- df_original %>%
          filter(aaSeqCDR3 %in% df_downsampled$CDR3.aa)
        
        output_filename <- paste0(output_dir, "/", type, "_downsampled_", original_filename)
        
        if (original_ext == "csv") {
          write.csv(final_df, file = output_filename, row.names = FALSE)
        } else if (original_ext %in% c("tsv", "txt")) {
          write.table(final_df, file = output_filename, sep = "\t", row.names = FALSE, quote = FALSE)
        }
        
        cat(" - Saved:", output_filename, "\n")
      }
    }
  }
  cat("All files processed! Results saved in '", output_dir, "'\n", sep="")
}