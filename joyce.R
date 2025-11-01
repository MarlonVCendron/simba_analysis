# --- 0) Pacotes ---------------------------------------------------
packs <- c(
  "tidyverse","readr","stringr","geepack","glmmTMB","DHARMa","statmod",
  "broom.mixed","emmeans","performance","MuMIn","cAIC4","moments"
)
inst <- packs[!packs %in% installed.packages()[,"Package"]]
if(length(inst)) install.packages(inst, repos = "https://cloud.r-project.org")
invisible(lapply(packs, library, character.only = TRUE))

# --- 0.1) Diretório e arquivo ------------------------------------
# Ajuste para seu ambiente:
setwd("D:/teste_glmm")
data_file <- "rearing.csv"

# --- 1) Leitura robusta e padronização ---------------------------
.read_auto <- function(path){
  first <- readLines(path, n = 1, warn = FALSE)
  counts <- c(semicolon = stringr::str_count(first, ";"),
              comma     = stringr::str_count(first, ","),
              tab       = stringr::str_count(first, "\t"))
  delimiters_map <- c(semicolon = ";", comma = ",", tab = "\t")
  if (max(counts) == 0) {
    delim <- ","
    warning("Nenhum delimitador (;, , ou \\t) encontrado na primeira linha. Usando ','.")
  } else {
    max_name <- names(counts)[which.max(counts)]
    delim <- delimiters_map[max_name]
  }
  dec_mark <- if (delim == ";") "," else "."
  readr::read_delim(
    path, delim = delim,
    locale = readr::locale(decimal_mark = dec_mark),
    show_col_types = FALSE, trim_ws = TRUE, guess_max = 10000
  )
}

raw <- .read_auto(data_file)
cat("Estrutura da base RAW:\n"); print(glimpse(raw))

# --- 1.1) Reestruturação Wide -> Long (foco em DUR_AVG_*) --------
base_long <- raw |>
  dplyr::rename(id = Video, grupo_raw = GRUPO) |>
  dplyr::mutate(
    grupo = factor(ifelse(grupo_raw %in% c(1, "1", "muscimol", "MUSCIMOL"), 1, 0),
                   levels = c(0,1), labels = c("Salina","Muscimol")),
    id = factor(id)
  ) |>
  tidyr::pivot_longer(
    cols = tidyselect::matches("^DUR_AVG_(T|S1|S2)$"),
    names_to      = c(".value", "sessao_raw"),
    names_pattern = "^(DUR_AVG)_(T|S1|S2)$",
    values_drop_na = TRUE
  ) |>
  dplyr::mutate(
    sessao  = factor(toupper(sessao_raw), levels = c("T","S1","S2")),
    DUR_AVG = suppressWarnings(readr::parse_number(as.character(DUR_AVG)))
  ) |>
  dplyr::select(id, grupo, sessao, rearing = DUR_AVG) |>
  dplyr::filter(!is.na(sessao)) |>
  dplyr::arrange(id, sessao, grupo)

base <- base_long
cat("\nEstrutura da base LONG (base para modelagem):\n"); print(glimpse(base))

# Exporta CSV “arrumado”
readr::write_csv(base, "rearing_long.csv")
cat("\nArquivo salvo: rearing_long.csv\n")

# --- 2) Descritivas e checagens rápidas --------------------------
desc <- base |>
  dplyr::group_by(grupo, sessao) |>
  dplyr::summarise(
    n     = dplyr::n(),
    media = mean(rearing, na.rm = TRUE),
    var   = var(rearing,  na.rm = TRUE),
    skew  = moments::skewness(rearing, na.rm = TRUE),
    .groups = "drop"
  )
cat("\nEstatísticas Descritivas (média/variância/skew):\n"); print(desc)

has_zero <- sum(base$rearing == 0, na.rm = TRUE) > 0
cat("\nZeros na variável rearing? ", has_zero, "\n")

# --- 3) Especificações de família e link -------------------------
fam_log <- Gamma(link = "log")
fam_id  <- Gamma(link = "identity")

# --- 4) GEE (marginal) -------------------------------------------
cat("\n========================= GEE (Marginal) =======================\n")
gee_log <- geeglm(rearing ~ grupo * sessao, id = id, data = base,
                  family = fam_log, corstr = "exchangeable")
gee_id  <- geeglm(rearing ~ grupo * sessao, id = id, data = base,
                  family = fam_id,  corstr = "exchangeable")

cat("\nGEE (Gamma + log):\n");   print(summary(gee_log))
cat("\nGEE (Gamma + identity):\n"); print(summary(gee_id))

# QIC/QICu (critério p/ GEE)
QIC_log <- geepack::QIC(gee_log)
QIC_id  <- geepack::QIC(gee_id)
cat("\nQIC/QICu GEE:\n"); print(data.frame(
  Modelo = c("GEE_log", "GEE_id"),
  QIC  = c(QIC_log["QIC"],  QIC_id["QIC"]),
  QICu = c(QIC_log["QICu"], QIC_id["QICu"])
))

# Teste de diferentes estruturas de correlação no GEE
cat("\nComparando estruturas de correlação (GEE, link=log):\n")
gee_exch <- gee_log
gee_ar1  <- geeglm(rearing ~ grupo*sessao, id=id, data=base, family=fam_log, corstr="ar1")
gee_unst <- geeglm(rearing ~ grupo*sessao, id=id, data=base, family=fam_log, corstr="unstructured")

cmp_cor <- data.frame(
  modelo = c("exchangeable","ar1","unstructured"),
  QIC    = c(geepack::QIC(gee_exch)["QIC"], geepack::QIC(gee_ar1)["QIC"], geepack::QIC(gee_unst)["QIC"]),
  QICu   = c(geepack::QIC(gee_exch)["QICu"], geepack::QIC(gee_ar1)["QICu"], geepack::QIC(gee_unst)["QICu"])
)
print(cmp_cor)

# Resíduos de Pearson e Q-Q simples para GEE
pearson <- residuals(gee_exch, type = "pearson")
png("gee_pearson_diagnostics.png", width=1200, height=500)
par(mfrow=c(1,2))
qqnorm(pearson, main="GEE exchangeable - QQ de resíduos de Pearson"); qqline(pearson, col=2)
plot(fitted(gee_exch), pearson, xlab="Ajustado", ylab="Resíduo de Pearson",
     main="GEE exchangeable - Pearson vs Ajustado"); abline(h=0, lty=2, col=2)
par(mfrow=c(1,1))
dev.off()
cat("Figura salva: gee_pearson_diagnostics.png\n")

# --- 5) GLMM (condicional) ---------------------------------------
cat("\n========================= GLMM (Condicional) ====================\n")
glmm_log <- glmmTMB(rearing ~ grupo * sessao + (1|id),
                    data = base, family = fam_log)
glmm_id  <- glmmTMB(rearing ~ grupo * sessao + (1|id),
                    data = base, family = fam_id)

cat("\nGLMM (Gamma log):\n"); print(summary(glmm_log))
cat("\nGLMM (Gamma identity):\n"); print(summary(glmm_id))

cat("\nAIC GLMM (menor melhor):\n"); print(AIC(glmm_log, glmm_id))

# Diagnósticos DHARMa (Q-Q/Uniforme, dispersão etc.)
res_log <- DHARMa::simulateResiduals(glmm_log, n = 1000)
png("glmm_log_DHARMa.png", width=900, height=900)
plot(res_log)
dev.off()
cat("Figura salva: glmm_log_DHARMa.png\n")
print(testResiduals(res_log))

res_id  <- DHARMa::simulateResiduals(glmm_id, n = 1000)
png("glmm_identity_DHARMa.png", width=900, height=900)
plot(res_id)
dev.off()
cat("Figura salva: glmm_identity_DHARMa.png\n")
print(testResiduals(res_id))

# --- 6) Efeitos fixos, EMMs e Contrastes --------------------------
efeitos_glmm <- broom.mixed::tidy(glmm_log, effects = "fixed") |>
  dplyr::mutate(
    RR     = exp(estimate),
    CI_low = exp(estimate - 1.96 * std.error),
    CI_hi  = exp(estimate + 1.96 * std.error)
  )
cat("\nEfeitos fixos (GLMM, Gamma log) com Razões (RR):\n"); print(efeitos_glmm)

emm_int <- emmeans::emmeans(glmm_log, ~ grupo * sessao, type = "response")
cat("\nEMMs (médias marginais, escala original):\n"); print(emm_int)

ct_grupo_dentro_sessao <- emmeans::contrast(emm_int, method = "revpairwise", by = "sessao")
cat("\nContrastes Muscimol vs Salina por sessão:\n"); print(ct_grupo_dentro_sessao)

ct_sessao_dentro_grupo <- emmeans::contrast(emm_int, method = "pairwise", by = "grupo")
cat("\nContrastes entre sessões dentro de cada grupo:\n"); print(ct_sessao_dentro_grupo)

# Exporta resultados
readr::write_csv(as.data.frame(emm_int), "emm_glmm_gamma_log.csv")
readr::write_csv(as.data.frame(ct_grupo_dentro_sessao), "contrastes_grupo_por_sessao.csv")
readr::write_csv(as.data.frame(ct_sessao_dentro_grupo), "contrastes_sessao_por_grupo.csv")
cat("\nArquivos salvos: emm_glmm_gamma_log.csv, contrastes_grupo_por_sessao.csv, contrastes_sessao_por_grupo.csv\n")

# --- 7) Validação adicional (LOCO: leave-one-cluster-out) ---------
get_ids <- levels(base$id)
loco <- lapply(get_ids, function(drop_id){
  fit <- try(suppressWarnings(
    glmmTMB(rearing ~ grupo*sessao + (1|id), data = subset(base, id != drop_id),
            family = fam_log)
  ), silent = TRUE)
  if(inherits(fit, "try-error")) {
    data.frame(id = drop_id, AIC = NA_real_)
  } else {
    data.frame(id = drop_id, AIC = AIC(fit))
  }
}) |> dplyr::bind_rows()

print(loco)
readr::write_csv(loco, "loco_glmm_gamma_log.csv")
cat("Arquivo salvo: loco_glmm_gamma_log.csv\n")

# --- 8) Gráficos simples (médias por grupo*sessao) ----------------
emm_df <- as.data.frame(emm_int)
p <- ggplot(emm_df, aes(x = sessao, y = response, group = grupo)) +
  geom_line(aes(linetype = grupo)) +
  geom_point(aes(shape = grupo), size = 3) +
  geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL), width = .1) +
  labs(x = "Sessão", y = "Rearing (média estimada, Gamma log)", linetype = "Grupo", shape = "Grupo") +
  theme_minimal(base_size = 12)
ggsave("emm_glmm_gamma_log_plot.png", p, width = 7, height = 5, dpi = 150)
cat("Figura salva: emm_glmm_gamma_log_plot.png\n")