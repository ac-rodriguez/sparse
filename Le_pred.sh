#!/bin/bash

set -x
set -e
models=(simpleA9)
# DATASET=kalimantan_barat
# tiles=(T48MZD T48NZF T48NZH T49MBU T49MBV T49MCS T49MCT T49MCU T49MCV T49MDS T49MDT T49MDU T49MDV T49MES T49MET T49MEU T49MEV T49MFV T49MGV T49NBA T49NBB T49NBC T49NCA T49NCB T49NCC T49NDA T49NDB T49NEA T49NEB T49NFA T49NFB T49NGA T49NGB T49NHA T49NHB)
# locs=(palmcountries_2017)
# weights=$WORK/sparse/training/snapshots/palm/palmsarawak3/simpleA9_drop/PATCH16_16_Lr0.5_sq2adam2.5e-4


# DATASET=palmborneo
#tiles=(T49MCT T49MCU T49MCV)
#tiles=(T48MZD T48NZF T48NZH T49MBU T49MBV T49MCS T49MCT T49MCU T49MCV T49MDS T49MDT T49MDU T49MDV T49MES T49MET T49MEU T49MEV T49MFS T49MFT T49MFU T49MFV T49MGS T49MGT T49MGU T49MGV T49MHR T49MHS T49MHT T49MHU T49MHV T49NBA T49NBB T49NBC T49NCA T49NCB T49NCC T49NDA T49NDB T49NEA T49NEB T49NFA T49NFB T49NGA T49NGB T49NHA T49NHB T50MKA T50MKB T50MKC T50MKD T50MKE T50MLA T50MLB T50MLC T50MLD T50MLE T50MLV T50MMA T50MMB T50MMC T50MMD T50MME T50MMV T50MNB T50MNC T50MND T50MNE T50MNV T50MPC T50NKF T50NKG T50NKH T50NKJ T50NLF T50NLG T50NLH T50NLJ T50NLK T50NMF T50NMG T50NMH T50NMJ T50NMK T50NNF T50NNG T50NNH T50NNJ T50NNK T50NPF T50NPG T50NPH T50NPK T50NQF T50NQG T50NQH T49NCB T49NCC T49NDA T49NDB T49NDC T49NEB T49NEC T49NED T49NFB T49NFC T49NFD T49NGB T49NGC T49NGD T49NGE T49NHB T49NHC T49NHD T49NHE T49NHF T50NKG T50NKH T50NKJ T50NKK T50NKL T50NLH T50NLJ T50NLK T50NLL T50NLM T50NMK T50NML T50NMM T50NMN T50NMP T50NNK T50NNL T50NNM T50NNN T50NNP T50NPK T50NPL T50NPM T50NPN T50NQK T50NQL T50NQM)
#tiles=(T48MZD T48NZF T48NZH T49MBU T49MBV T49MCS T49MDS T49MDT T49MDU T49MDV T49MES T49MET T49MEU T49MEV T49MFS T49MFT T49MFU T49MFV T49MGS T49MGT T49MGU T49MGV T49MHR T49MHS T49MHT T49MHU T49MHV T49NBA T49NBB T49NBC T49NCA T49NCB T49NCC T49NDA T49NDB T49NEA T49NEB T49NFA T49NFB T49NGA T49NGB T49NHA T49NHB T50MKA T50MKB T50MKC T50MKD T50MKE T50MLA T50MLB T50MLC T50MLD T50MLE T50MLV T50MMA T50MMB T50MMC T50MMD T50MME T50MMV T50MNB T50MNC T50MND T50MNE T50MNV T50MPC T50NKF T50NKG T50NKH T50NKJ T50NLF T50NLG T50NLH T50NLJ T50NLK T50NMF T50NMG T50NMH T50NMJ T50NMK T50NNF T50NNG T50NNH T50NNJ T50NNK T50NPF T50NPG T50NPH T50NPK T50NQF T50NQG T50NQH T49NCB T49NCC T49NDA T49NDB T49NDC T49NEB T49NEC T49NED T49NFB T49NFC T49NFD T49NGB T49NGC T49NGD T49NGE T49NHB T49NHC T49NHD T49NHE T49NHF T50NKG T50NKH T50NKJ T50NKK T50NKL T50NLH T50NLJ T50NLK T50NLL T50NLM T50NMK T50NML T50NMM T50NMN T50NMP T50NNK T50NNL T50NNM T50NNN T50NNP T50NPK T50NPL T50NPM T50NPN T50NQK T50NQL T50NQM)
#locs=(palmcountries_2017)
#weights=$WORK/sparse/training/palm/palmborneo/simpleA9_drop_wloc/PATCH16_16_Lr0.5_sq2a
#weights=$WORK/sparse/training/palm/palmborneo/simpleA30_drop/PATCH16_16_Lr0.5_sq2adam2.5e-4

#DATASET=coco_sulawesi
# 82
#tiles=(T50MNV T50MPB T50MPC T50MPU T50MPV T50MQA T50MQB T50MQC T50MQD T50MQE T50MQU T50MQV T50MRA T50MRB T50MRC T50MRD T50MRE T50MRT T50MRU T50MRV T50NQF T50NRF T50NRG T51MTM T51MTN T51MTP T51MTQ T51MTR T51MTS T51MTT T51MTU T51MTV T51MUM T51MUN T51MUP T51MUQ T51MUR T51MUS T51MUT T51MUU T51MUV T51MVN T51MVP T51MVQ T51MVR T51MVS T51MVT T51MVU T51MVV T51MWP T51MWQ T51MWR T51MWS T51MWT T51MWU T51MWV T51MXP T51MXQ T51MXT T51MXU T51NTA T51NTB T51NUA T51NUB T51NVA T51NVB T51NWA T51NWB T51NXA T51NXB T51NXC T51NYB T51NYC T51NYD T51NYE T51NYF T51NZD T51NZE T51NZG T52NBK T52NBL T52NBM)
#locs=(palmcountries_2017)
#weights=$WORK/sparse/training/palm/palmborneo/simpleA9_drop/PATCH16_16_Lr0.5_sq2adam2.5e-4
#weights=$WORK/sparse/training/coco/cocopalawanplus/simpleA9_drop/PATCH16_16_Lr0.5_sq2adam2.5e-4

# DATASET=cocopreactive
#117
# tiles=(T50NMP T50NNP T50NPM T50NPN T50NPP T50NQL T50NQM T50NRL T50NRM T50NRN T50PMQ T50PNQ T50PNR T50PPQ T50PPR T50PPS T50PQC T50PQR T50PQT T50PQU T50PRA T50PRB T50PRC T50PRS T50PRT T50PRU T50PRV T50QQD T50QRD T50QRE T50QRF T51NTF T51NTG T51NTH T51NUG T51NUH T51NUJ T51NVH T51NVJ T51NWH T51NWJ T51NXG T51NXJ T51NYF T51NYG T51NYH T51NYJ T51NZG T51NZH T51NZJ T51PTL T51PTM T51PTN T51PTP T51PTQ T51PTR T51PTS T51PTT T51PUL T51PUM T51PUN T51PUQ T51PUR T51PUS T51PUT T51PVK T51PVL T51PVM T51PVN T51PVP T51PVQ T51PVR T51PVS T51PVT T51PWK T51PWL T51PWM T51PWN T51PWP T51PWQ T51PWR T51PWS T51PXK T51PXL T51PXM T51PXN T51PXP T51PXQ T51PXR T51PYK T51PYL T51PYM T51PYN T51PYP T51PYQ T51PZK T51PZL T51PZM T51PZN T51QTA T51QTU T51QTV T51QUA T51QUB T51QUC T51QUD T51QUU T51QUV T51QVA T51QVB T51QVC T51QVU T51QVV T52NBN T52NBP T52PBQ T52PBR)  # T50PQS T51PUP T51NXH
#tiles=(T50PQS T51PUP T51NXH)
# locs=(phillipines_2017)
#weights=$WORK/sparse/training/palm/palmborneo/simpleA9_drop/PATCH16_16_Lr0.5_sq2adam2.5e-4
#weights=$WORK/sparse/training/coco/cocopalawanplus/simpleA9_drop/PATCH16_16_Lr0.5_sq2adam2.5e-4
# weights=$WORK/sparse/training/coco/cocopreactive_june2020/simpleA9_drop_wloc/PATCH16_16_Lr0.5_sq2b
# weights=$WORK/sparse/training/coco/cocopreactive/simpleA9_drop_wloc/PATCH16_16_Lr0.5_sq2



# DATASET=palm4748a
# # 163
# tiles=(T47MLV T47MMV T47MNS T47MNT T47MNU T47MNV T47MPS T47MPT T47MPU T47MQR T47MQS T47MQT T47MQV T47MRP T47MRQ T47MRR T47MRT T47NLD T47NLB T47NKA T47NKB T47NKC T47NKD T47NKE T47NLA T47NLC T47NLF T47NMA T47NMB T47NPC T47NME T47NMF T47NNE T47NPB T47NPD T47NQA T47NQB T47NQC T47NRB T47NRC T48MTA T48MTB T48MUB T47NND T48MTE T48MTU T48MTV T47NNC T48MUD T48MUE T48MUU T48MUV T48MVA T48MVC T48MVD T48MVE T48MVU T48NUG T48MWA T48MWC T48MWD T48MWE T48MWT T48MWU T48MXA T48MXB T48MXC T48MXD T48MXS T48MXT T48MXU T48MXV T48MYB T48MYC T48MYS T48MYT T48MYU T48MZA T48MZB T48MZC T48MZD T48MZS T48MZT T48MZU T48NTF T48NTG T48NUF T48NUG T48NVF T48NVG T48NWH T48NWJ T48NXG T48NXH T48NXJ T48NYF T48NYG T48NYJ T48NYK T48NYL T48NZF T48NZH T48NZJ T48NZK T48NZL T47NNA T47MMU T48MVV T47NLE T48NUG T48NTK T47NNG T47NNH T47NPE T48NTK T48MWV T47NPG T47NPH T47NQC T48MWB T48MVB T47NQE T47NQF T47NQG T47NQH T47NRC T47NRD T47NRE T47NRF T48MUC T48MUA T47NRH T48NTG T48NTH T48NTJ T48MTD T48MTC T48NTL T48NTM T47NRG T47NRG T48NUH T48NUJ T48NUK T48NUL T48NVG T48NVH T48NVJ T47NRA T47NQD T47NQD T47NPF T47NPF T47NPA T47NMD T47NMC T47NKF T47MRU T47MRS T47MQU T47MPV T47MRV T47NNB)
# # tiles=(T48MVV T48NUG T47NNC T47NNA T47NQE T46NGL T47NLE T48MTD T48NTJ T48MVB)
# locs=(palmcountries_2017)
# # weights=$WORK/sparse/training/palm/palm4748/simpleA9_drop_wloc/PATCH16_16_Lr0.5_sq2
# #weights=$WORK/sparse/training/palm/palm4748a/simpleA9_drop_wloc/PATCH16_16_Lr0.5_sq2/last
# weights=$WORK/sparse/training/palm/palm4748a/simpleA9_wloc/PATCH16_16_Lr0.5_sq2*/last




# YEAR=_2017
YEAR=_2020
DATASET=palm4_act
locs=(palmcountries_2017)
weights=$WORK/sparse/training/palm/palm4_act/simpleA9_wloc/PATCH16_16_Lr0.5_sq2adam*/last
# # 665
# # tiles=(T46NGH T46NGJ T46NGK T49NCB T46NGM T46NHH T46NHJ T46NHK T46NHM T47MLV T47MMV T47MNS T47MNT T47MNU T47MNV T47MPS T47MPT T47MPU T51NUB T47MQR T47MQS T47MQT T49NGD T47MQV T47MRP T47MRQ T47MRR T49NFD T47MRT T47NLD T47NLB T47NKA T47NKB T47NKC T47NKD T47NKE T51NXB T47NLA T51NXA T47NLC T50MMC T49NCB T47NLF T47NMA T47NMB T47NPC T50NQL T47NME T47NMF T50MMD T50MLC T49NHE T49MFT/49MGT T47NNE T49MEV/49NEA T47NPB T49MET T47NPD T47NQA T47NQB T47NQC T46NGL T47NRB T47NRC T48MTA T48MTB T48MUB T47NND T48MTE T48MTU T48MTV T47NNC T50NLM T49NED T48MUD T48MUE T48MUU T48MUV T48MVA T49NEC T48MVC T48MVD T48MVE T48MVU T48NUG T48MWA T48MWC T48MWD T48MWE T48MWT T48MWU T48MXA T48MXB T48MXC T48MXD T48MXS T48MXT T48MXU T48MXV T48MYB T48MYC T48MYS T48MYT T48MYU T48MZA T48MZB T48MZC T48MZD T48MZS T48MZT T48MZU T48NTF T48NTG T48NUF T48NUG T48NVF T48NVG T48NWH T48NWJ T48NXG T48NXH T48NXJ T48NYF T48NYG T48NYJ T48NYK T48NYL T48NZF T48NZH T48NZJ T48NZK T48NZL T49LDL T49LEL T49LFL T49LGL T49LHL T49MBM T49MBN T49MBP T49MBS T49MBT T49MBU T49MBV T49MCM T49MCN T49MCP T49MCS T49MCT T49MCU T49MCV T49MDM T49MDN T49MDP T49MDS T49MDT T49MDU T49MEM T49MEN T49MEP T49MES T49MEV T49MFM T49MFN T49MFP T49MFT T49MFU T49MFV T49MGM T49MGN T49MGS T49MGT T49MGU T49MGV T49MHM T49MHN T49MHP T49MHQ T49MHR T51MTT T49MHT T49MHU T49MHV T49NBA T49NBB T49NBC T49NBD T49NBE T49NCA T47NNA T49NCC T49NDA T49NDB T49NEA T49NEB T47MMU T49NFB T50NLL T49NGB T49NHA T49NHB T50LKR T50LLR T50LMQ T50LMR T50LNQ T50LNR T50LPR T50LQQ T50LQR T50LRP T50LRQ T50LRR T50MKA T50MKB T50MKC T50MKD T50MKE T50MKS T50MKT T50MKU T50MKV T50MLA T50MLB T50MLC T50MLD T50MLE T50MLS T50MLT T50MLV T50MMA T50MMB T50MMC T50MMD T49NDB T50MMS T50MMT T50MMV T50MNB T50MNC T50MND T50MNE T50MNS T50MNT T50MNV T50MPB T50MPC T50MPS T50MPT T50MPU T50MPV T50MQA T50MQB T52MGE T50MQE T50MQS T50MQT T50MQU T50MQV T50MRA T50MRB T50MRC T50MRE T50MRT T50MRU T50MRV T50NKF T50NKG T50NKH T50NKJ T50NLF T50NLG T50NLH T50NLJ T50NLK T50NMF T50NMG T50NMH T50NMJ T50NMK T50NNF T50NNG T50NNH T50NNJ T50NNK T50NPF T49NCA T50NPH T50NPK T50NQF T50NQG T50NQH T50NRF T50NRG T48MVV T47NLE T54MVS T51LTJ T51LTK T51LTL T51LUJ T51LUL T51LVH T51LVJ T51LVL T51LWH T51LWJ T51LWK T51LWL T51LXJ T51LXK T51LXL T51LYK T51LYL T51LZL T51MTM T51MTN T51MTP T51MTQ T51MTR T51MTS T52MGD T51MTU T51MTV T51MUM T51MUN T51MUP T51MUQ T51MUS T51MUT T51MUU T51MUV T51MVM T51MVN T51MVP T51MVQ T51MVT T51MVV T51MWM T51MWP T51MWQ T51MWR T51MWS T51MWT T51MWU T51MWV T51MXM T51MXP T51MXQ T51MXT T51MXU T51MYM T51MYT T51MYU T51MZM T51MZN T51MZR T51MZS T51MZT T51MZU T51NTA T51NTB T51NUA T51NVA T51NVB T51NWA T51NWB T51MVU T51NXC T51MVR T51MUR T50NQL T51NYC T51NYD T51NYE T51NYF T50NQL T51NZB T51NZD T51NZE T51NZG T50NPG T50NNK T50NNK T50NNF T50NMN T50NMN T50NLM T50NLL T50MRD T50MQD T50MQC T49NHE T49NHE T49NGA T49NFA T49NED T49NEC T49NEB T49NEB T52LCR T52LDR T52LER T52LFR T52LGR T52MBA T52MBB T52MBC T52MBD T52MBE T52MBS T52MBT T52MCA T52MCB T52MCC T52MCD T52MCE T52MCS T52MCU T52MCV T52MDA T52MDB T52MDC T52MDD T52MDE T52MDS T52MDT T52MEA T52MEB T52MEC T52MED T52MEE T52MES T52MET T52MEU T52MEV T52MFA T52MFB T52MFC T52MFD T52MFE T52MFS T52MFU T52MFV T52MGA T52MGB T52MGC T49NDB T49NCB T52MGS T52MGT T52MGV T52MHA T52MHB T52MHC T49MHS T52MHE T52MHS T52MHT T52MHU T52MHV T52NBF T52NBG T52NBK T52NBL T52NBM T52NCF T52NCG T52NCH T52NDF T52NDG T52NDH T52NDJ T52NEF T52NFF T52NGF T52NGG T53LQL T53LRL T53MKP T53MKQ T53MKR T53MKS T53MKT T53MKU T53MKV T53MLN T53MLP T49MFS T53MLS T53MLT T53MLU T49MEU T53MMN T53MMP T53MMQ T53MMR T53MMS T53MMT T53MMU T53MMV T53MNQ T53MNR T53MNS T53MNT T53MNU T53MNV T53MPQ T49MET T53MPS T49MDV T53MPU T53MPV T53MQM T53MQQ T53MQR T53MQS T53MQT T53MQU T53MRM T53MRN T53MRP T53MRQ T53MRR T53MRS T53MRT T53MRU T53NLA T53NMA T53NMB T54LTR T54LUR T54LVQ T54LVR T54LWQ T54LWR T54MTA T54MTB T54MTC T54MTD T54MTS T54MTT T54MTU T54MTV T54MUA T54MUB T48NUG T54MUS T54MUT T54MUU T54MUV T54MVA T54MVB T54MVC T48NTK T54MVT T54MVU T54MVV T54MWA T54MWB T54MWC T54MWS T54MWU T54MWV T47NNG T47NNH T47NPE T48NTK T48MWV T47NPG T47NPH T47NQC T48MWB T48MVB T47NQE T47NQF T47NQG T47NQH T47NRC T47NRD T47NRE T47NRF T48MUC T48MUA T47NRH T48NTG T48NTH T48NTJ T48MTD T48MTC T48NTL T48NTM T47NRG T47NRG T48NUH T48NUJ T48NUK T48NUL T48NVG T48NVH T48NVJ T49NCB T49NCC T49NDA T47NRA T47NQD T49NDC T47NQD T47NPF T47NPF T47NPA T47NMD T47NMC T49NFB T49NFC T49NFD T49NGB T49NGC T49NGD T49NGE T49NHB T49NHC T49NHD T47NKF T47MRU T49NHF T50NKG T50NKH T50NKJ T50NKK T50NKL T50NLH T50NLJ T50NLK T47MRS T47MQU T47MPV T53MPT T50NMK T50NML T50NMM T50NMP T51MVS T50NPM/50NNM T50NNL T50NNM T50NNN T50NNP T50NPK T50NPL T50NPM T50NPN T50NQK T50NPL/50NPK T50MME T50NQM T47MRV T54MUC T53MPR T53MLV T53MLR T52MHD T51NYB T47NNB T46NHL)
# # tiles=(T48MVV T48NUG T47NNC T47NNA T47NQE T46NGL T47NLE T48MTD T48NTJ T48MVB)
# # weights=$WORK/sparse/training/palm/palm4748/simpleA9_drop_wloc/PATCH16_16_Lr0.5_sq2
# #weights=$WORK/sparse/training/palm/palm4748a/simpleA9_drop_wloc/PATCH16_16_Lr0.5_sq2/last
# #weights=$WORK/sparse/training/palm/palm4/simpleA9_wloc/PATCH16_16_Lr0.5_sq211sept*/last
tiles=(T47MRS T47NLC T49LHL T49MGS)
# tiles=(T47NPE T49MHN T52MBE T51NXB T50NRG T50MQE T47NKB T47MMU T51LVL T48NVG T50NMF T47MRS T47NLC T49LHL T49MGS)

# for asia_2019
# YEAR=_2019
# DATASET=palm2019

# locs=(asia_2019)
# # # 555
# # # tiles=(T48MTD T52NBG T48MUB T52MFU T50NMH T54MWB T50MRE T49NBE T47MMV T51MWT T49MDP T54MVS T49MDN T52MEB T52MHB T47MQS T47MQT T48MZA T49MGV T48NVF T53LRL T50MQD T47MPS T48MWA T52MBA T49MCP T51LUJ T54LWQ T53MMV T51MVP T48NVG T51MYU T53MPU T50NKF T50MRA T52MGD T49MFV T52MHC T47NNB T51NZD T52MHU T51LWL T49MGN T47MNV T47NNA T51MVQ T48NWH T48MZU T50MLC T54MTA T48MUE T49MCS T51NVA T48MWT T49MCV T49MCM T54MTC T50MPB T50MQC T46NGL T49NHB T50LKR T50MRC T52MHT T48MWC T52MFS T52MDA T54MUT T51MWR T51MZS T50NLK T47NNE T51NUA T49NBB T52MDC T54MWC T50MKT T49NDB T54MUS T47MRV T50NQF T51LTL T52MFV T53MKQ T49MET T52MES T53MRP T54MVC T50MKV T50MPU T53MQS T50MQU T53MMN T50MRT T49MBP T51LYL T47NMD T52MEC T47NKD T50NNG T51MWV T50MQV T47MQV T46NHM T51LVH T53MNS T51MTR T48NZJ T48MYC T51MTN T51LVJ T50NMF T53MLR T49MGU T49LDL T53MQU T54MTB T47NQB T49NHA T51MXU T47MRU T49MFS T52MCE T52NCH T49NBC T54MTD T48MYU T51MVN T47MPU T50NNH T48MWE T49MBM T52LER T47NPB T52MCB T51NTB T51LTJ T48NWJ T47NMB T48MTA T49MFN T49LGL T51MWP T48MXD T53MRR T53MLS T49MGS T48NXG T50LQQ T50NPF T49NDA T48MVB T49LEL T50MNE T48NTF T47MRS T48MTU T49MEU T54MVT T48MVU T48MUA T50MMC T50MRV T51LYK T50MNT T52MDT T51NZB T50MLE T50MLB T52MBC T54MWS T53MNT T52LCR T51NYB T52NCF T48MTE T53MKV T52MFB T47NRC T49MHQ T50NNJ T53MQT T54MUA T49MDV T51MXQ T49MEN T53MQR T54MVB T50NMJ T52LDR T49MFT T47MRQ T51MVV T50NNF T51MZM T51MXM T50MLS T53MNV T51MVR T48NZL T48MVV T46NHJ T49MHT T51NUB T49MHS T49MFU T54MTV T50MKE T52MBB T53MRU T53MRQ T52MHE T53MMS T51MUN T50MMV T47NKE T48MUU T50MQA T49NFA T47MRR T50MPC T48NYL T48MZC T53NMA T49MCU T50MKS T54MWU T50NPG T52MFE T51MWQ T52MDD T49NEA T50MRU T51NWA T54LVR T50MNC T52MBS T51MWS T51MVM T52MCC T53MKP T52MCU T53MNQ T54LUR T53MMP T52NBL T52NDH T51MZU T50NQG T54MUC T48MXC T49MBV T54MVA T52MGV T53MKT T47MQR T48MTV T53MKU T51NVB T47NNC T53MPR T51MVS T48NYK T53MQM T48MWD T50MND T52MGS T51MUM T49MCT T48MVD T47NPC T50LMQ T52NEF T53NMB T50MRB T50NKH T50NQH T50NLJ T49LFL T47NME T47MRP T50NPK T51NXC T52MBE T49MES T51MUQ T47NLB T53MNR T48MUC T53MPV T52NGF T53MLT T51MTV T51MVU T51MTQ T50MQE T49NGB T48MVC T50MKA T52MHV T47MNT T52MDE T52NDG T52MEA T53MLP T49NCC T47NPA T47NND T47MRT T49MEV T48NTG T50MMD T48MWV T52NDF T51MUR T51NYF T53MRT T50MNV T52NCG T51MTM T49NCB T50MLD T51MTU T48MYT T51MWM T50NKG T51MUP T51MZT T48MXU T51LWH T50MQS T48MZD T52MFC T48NYF T50MME T46NHH T49LHL T51MZN T47NMA T51MYM T48MVA T47MPV T51LXJ T53MQQ T54MWA T47NMF T51MUS T54MVU T51MUU T49MBU T50MLA T52MDS T54LTR T48MZT T49MFM T54LVQ T52MCA T52MFD T46NGH T48MVE T52LFR T51LUL T51MUV T48MYB T48NZF T51MUT T50MMT T48MUV T50NRG T49MGT T48MXV T51NXB T52MCV T48MZB T51NYE T48MXT T46NHL T50MRD T47MPT T51LTK T52MEV T52MGC T48MWU T48NYG T52NBF T50MMA T52MFA T50LRR T47MNU T47NKA T47NLA T51NYD T49MHM T51MTP T52MHA T47MQU T47MLV T50MNS T51MXP T52NGG T50NLF T48NZH T52MHD T48NZK T51MWU T50MQT T49MHV T49MBN T53MLN T47MMU T52MGB T48NYJ T50MPT T52NBK T49MBT T48MXB T46NHK T50MMS T49NGA T51NZE T53MLV T47NKB T51MVT T51MZR T53MRN T53MKR T47NQC T51MXT T48MTB T50LPR T49NBD T47NMC T49MHU T51NYC T53MRM T47NRB T50MLT T51LXK T47NKF T47NQA T51LWK T54MUV T47NLF T48NUG T54MUB T52MEU T47NLC T46NGJ T52MBD T50LRP T50MKB T49MDS T50MLV T51LXL T49MDM T46NGM T52MEE T53MNU T50NLG T50NMG T51LZL T48MXS T50MKD T50NMK T51MTS T52MBT T49MFP T47NLD T48NUF T52MGT T49MDU T51NWB T47NRA T53MMT T50MPS T50LMR T54MTT T48MZS T54MWV T50MPV T49MHN T50NKJ T50LNR T52MDB T49MDT T48NXJ T49MCN T50MKU T50NLH T50LRQ T51MTT T50LQR T49MHP T49NFB T47NLE T50NPH T48MYS T48MTC T49MEP T47NPD T47MNS T52MCS T48MXA T54MVV T54MTU T50MQB T49MHR T52MHS T50NNK T53MLU T50LNQ T49NBA T51NTA T50MNB T51LWJ T48MUD T52MGE T53MPT T52LGR T50NRF T49MEM T51LVL T53MRS T52NFF T50MMB T53MMU T46NGK T48MWB T52MED T53MPS T52NBM T50MKC T53MMQ T51NZG T47NKC T52NDJ T49NCA T49MBS T53NLA T53MPQ T48NXH T54LWR T53MMR T54MUU T52MGA T54MTS T49MGM T53MKS T51NXA T53LQL T49NEB T50LLR T51MYT T52MET T52MCD)
# # # 79
# # # tiles=(T47NQC T49NFD T50NKH T50NLJ T47NPE T49NDA T48NVH T48NTK T50NPK T48NUH T47NPH T48NUG T50NNN T47NQG T49NGE T49NEC T49NGB T47NQD T49NGC T48NVG T47NRE T49NCC T50NMK T48NTG T50NQM T49NHD T47NRC T50NNP T49NCB T48NUJ T48NVJ T50NKJ T49NFC T48NTJ T50NLH T50NKG T49NHF T50NNM T50NQK T47NRH T49NHB T48NTM T49NFB T49NGD T50NLK T48NUK T49NDB T50NNL T50NNK T50NKL T49NHC T47NRG T48NTH T50NPL T47NRF T49NDC T47NNH T47NNG T50NPM T50NLL T50NMN T47NPG T47NQE T48NTL T50NML T48NUL T50NQL T49NED T47NPF T47NQF T47NRD T50NKK T47NQH T50NLM T50NMM T49NHE T49NEB T50NMP T50NPN)
# # # 612
# tiles=(T46NGH T46NGJ T46NGK T46NGL T46NGM T46NHH T46NHJ T46NHK T46NHL T46NHM T47MLV T47MMU T47MMV T47MNS T47MNT T47MNU T47MNV T47MPS T47MPT T47MPU T47MPV T47MQR T47MQS T47MQT T47MQU T47MQV T47MRP T47MRQ T47MRR T47MRS T47MRT T47MRU T47MRV T47NKA T47NKB T47NKC T47NKD T47NKE T47NKF T47NLA T47NLB T47NLC T47NLD T47NLE T47NLF T47NMA T47NMB T47NMC T47NMD T47NME T47NMF T47NNA T47NNB T47NNC T47NND T47NNE T47NNG T47NNH T47NPA T47NPB T47NPC T47NPD T47NPE T47NPF T47NPG T47NPH T47NQA T47NQB T47NQC T47NQD T47NQE T47NQF T47NQG T47NQH T47NRA T47NRB T47NRC T47NRD T47NRE T47NRF T47NRG T47NRH T48MTA T48MTB T48MTC T48MTD T48MTE T48MTU T48MTV T48MUA T48MUB T48MUC T48MUD T48MUE T48MUU T48MUV T48MVA T48MVB T48MVC T48MVD T48MVE T48MVU T48MVV T48MWA T48MWB T48MWC T48MWD T48MWE T48MWT T48MWU T48MWV T48MXA T48MXB T48MXC T48MXD T48MXS T48MXT T48MXU T48MXV T48MYB T48MYC T48MYS T48MYT T48MYU T48MZA T48MZB T48MZC T48MZD T48MZS T48MZT T48MZU T48NTF T48NTG T48NTH T48NTJ T48NTK T48NTL T48NTM T48NUF T48NUG T48NUH T48NUJ T48NUK T48NUL T48NVF T48NVG T48NVH T48NVJ T48NWH T48NWJ T48NXG T48NXH T48NXJ T48NYF T48NYG T48NYJ T48NYK T48NYL T48NZF T48NZH T48NZJ T48NZK T48NZL T49LDL T49LEL T49LFL T49LGL T49LHL T49MBM T49MBN T49MBP T49MBS T49MBT T49MBU T49MBV T49MCM T49MCN T49MCP T49MCS T49MCT T49MCU T49MCV T49MDM T49MDN T49MDP T49MDS T49MDT T49MDU T49MDV T49MEM T49MEN T49MEP T49MES T49MET T49MEU T49MEV T49MFM T49MFN T49MFP T49MFS T49MFT T49MFU T49MFV T49MGM T49MGN T49MGS T49MGT T49MGU T49MGV T49MHM T49MHN T49MHP T49MHQ T49MHR T49MHS T49MHT T49MHU T49MHV T49NBA T49NBB T49NBC T49NBD T49NBE T49NCA T49NCB T49NCC T49NDA T49NDB T49NDC T49NEA T49NEB T49NEC T49NED T49NFA T49NFB T49NFC T49NFD T49NGA T49NGB T49NGC T49NGD T49NGE T49NHA T49NHB T49NHC T49NHD T49NHE T49NHF T50LKR T50LLR T50LMQ T50LMR T50LNQ T50LNR T50LPR T50LQQ T50LQR T50LRP T50LRQ T50LRR T50MKA T50MKB T50MKC T50MKD T50MKE T50MKS T50MKT T50MKU T50MKV T50MLA T50MLB T50MLC T50MLD T50MLE T50MLS T50MLT T50MLV T50MMA T50MMB T50MMC T50MMD T50MME T50MMS T50MMT T50MMV T50MNB T50MNC T50MND T50MNE T50MNS T50MNT T50MNV T50MPB T50MPC T50MPS T50MPT T50MPU T50MPV T50MQA T50MQB T50MQC T50MQD T50MQE T50MQS T50MQT T50MQU T50MQV T50MRA T50MRB T50MRC T50MRD T50MRE T50MRT T50MRU T50MRV T50NKF T50NKG T50NKH T50NKJ T50NKK T50NKL T50NLF T50NLG T50NLH T50NLJ T50NLK T50NLL T50NLM T50NMF T50NMG T50NMH T50NMJ T50NMK T50NML T50NMM T50NMN T50NMP T50NNF T50NNG T50NNH T50NNJ T50NNK T50NNL T50NNM T50NNN T50NNP T50NPF T50NPG T50NPH T50NPK T50NPL T50NPM T50NPN T50NQF T50NQG T50NQH T50NQK T50NQL T50NQM T50NRF T50NRG T51LTJ T51LTK T51LTL T51LUJ T51LUL T51LVH T51LVJ T51LVL T51LWH T51LWJ T51LWK T51LWL T51LXJ T51LXK T51LXL T51LYK T51LYL T51LZL T51MTM T51MTN T51MTP T51MTQ T51MTR T51MTS T51MTT T51MTU T51MTV T51MUM T51MUN T51MUP T51MUQ T51MUR T51MUS T51MUT T51MUU T51MUV T51MVM T51MVN T51MVP T51MVQ T51MVR T51MVS T51MVT T51MVU T51MVV T51MWM T51MWP T51MWQ T51MWR T51MWS T51MWT T51MWU T51MWV T51MXM T51MXP T51MXQ T51MXT T51MXU T51MYM T51MYT T51MYU T51MZM T51MZN T51MZR T51MZS T51MZT T51MZU T51NTA T51NTB T51NUA T51NUB T51NVA T51NVB T51NWA T51NWB T51NXA T51NXB T51NXC T51NYB T51NYC T51NYD T51NYE T51NYF T51NZB T51NZD T51NZE T51NZG T52LCR T52LDR T52LER T52LFR T52LGR T52MBA T52MBB T52MBC T52MBD T52MBE T52MBS T52MBT T52MCA T52MCB T52MCC T52MCD T52MCE T52MCS T52MCU T52MCV T52MDA T52MDB T52MDC T52MDD T52MDE T52MDS T52MDT T52MEA T52MEB T52MEC T52MED T52MEE T52MES T52MET T52MEU T52MEV T52MFA T52MFB T52MFC T52MFD T52MFE T52MFS T52MFU T52MFV T52MGA T52MGB T52MGC T52MGD T52MGE T52MGS T52MGT T52MGV T52MHA T52MHB T52MHC T52MHD T52MHE T52MHS T52MHT T52MHU T52MHV T52NBF T52NBG T52NBK T52NBL T52NBM T52NCF T52NCG T52NCH T52NDF T52NDG T52NDH T52NDJ T52NEF T52NFF T52NGF T52NGG T53LQL T53LRL T53MKP T53MKQ T53MKR T53MKS T53MKT T53MKU T53MKV T53MLN T53MLP T53MLR T53MLS T53MLT T53MLU T53MLV T53MMN T53MMP T53MMQ T53MMR T53MMS T53MMT T53MMU T53MMV T53MNQ T53MNR T53MNS T53MNT T53MNU T53MNV T53MPQ T53MPR T53MPS T53MPT T53MPU T53MPV T53MQM T53MQQ T53MQR T53MQS T53MQT T53MQU T53MRM T53MRN T53MRP T53MRQ T53MRR T53MRS T53MRT T53MRU T53NLA T53NMA T53NMB T54LTR T54LUR T54LVQ T54LVR T54LWQ T54LWR T54MTA T54MTB T54MTC T54MTD T54MTS T54MTT T54MTU T54MTV T54MUA T54MUB T54MUC T54MUS T54MUT T54MUU T54MUV T54MVA T54MVB T54MVC T54MVS T54MVT T54MVU T54MVV T54MWA T54MWB T54MWC T54MWS T54MWU T54MWV)

others=(
    # '--numpy-seed=1 --is-dropout-uncertainty --mc-repetitions=5 --tag=soft_mc5 --is-use-location --fusion-type=soft'
    '--numpy-seed=1 --tag=soft_ens5 --compression=12 --mc-repetitions=5 --is-use-location --fusion-type=soft'
)
# tag=westkalim

EXTRAARGS=''

#BSUB -W 4:00
#BSUB -o /cluster/scratch/andresro/sparse/output/predict_4_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/sparse/output/predict_4_Le.%J.%I.txt
#BSUB -R "rusage[mem=64000,ngpus_excl_p=1]"
#BSUB -n 1
##BSUB -N
#BSUB -J pred_palm[1-45]
##BSUB -u andresro@ethz.ch
 
#### BEGIN #####

index=$((LSB_JOBINDEX-1))

set +x
i=0
  for mod in "${models[@]}"; do
   for loc in "${locs[@]}"; do
       for tile in "${tiles[@]}"; do
       for oth in "${others[@]}"; do
           filelist=($WORK/barry_palm/data/2A/$loc/*${tile}*${YEAR}*.SAFE)
        #    filelist=($WORK/barry_palm/data/2A/$loc/*${YEAR}*${tile}*.SAFE)
           for file in "${filelist[@]}"; do
        #    echo $file
	       if [ "$index" -eq "$i" ]; then
                       set -x
	               MODEL=$mod
	               DATA=$file
                   OTHER=$oth
	               set +x
               fi
	       ((i+=1))
done;done;done;done;done;
echo $i combinations
echo $DATA , $MODEL
if [ -z "$LSB_JOBINDEX" ]
then
	echo "no LSB_JOBINDEX defined"
	exit 0
fi

#MODEL=${models[0]}
#TILE=${tiles[0]}
# VERSION=${dataset}_${MODEL}${tag}


set -x

PATCH=150 # ${patches[index]}
BATCH=10
BORDER=20
#DATALIST=($WORK/barry_palm/data/2A/$loc/*${TILE}*2017*.SAFE)

#echo ${#DATALIST[@]} files x ${#models[@]} datasets = $((${#DATALIST[@]}*${#models[@]}))

#DATA=${DATALIST[index]}
#if [ "$index" -gt "$i" ]; then
#exit 1;fi

#LAMW=${lambdas[index]}
LAMSR=0

set +x

# module load python_gpu/3.7.1  gdal/2.2.2 
# module load python_gpu/3.7.1  gdal/2.3.2  cudnn/7.6.4
# module load python_gpu/3.7.1  gdal/2.4.4  cudnn/7.6.4 # this is working for gdal2.x

module load python_gpu/3.7.1  gdal/3.1.2  cudnn/7.6.4


set -x


cd $HOME/code/sparsem
python3 -u predict.py \
    $DATA $weights \
    --model=$MODEL \
    --patch-size-eval=$PATCH \
    --border=$BORDER \
    --not-save-arrays \
    --batch-size-eval=$BATCH \
    --dataset=$DATASET \
    --save-dir=$WORK/sparse/inference $OTHER $EXTRAARGS 
    # --is-overwrite-pred
    # --tag=$VERSION  \

#
#### END #####


# for stats on a job
# grep -E -- 'Memory :|time :' predict_4_Le.10029785.*txt >> $HOME/stats10029785.txt