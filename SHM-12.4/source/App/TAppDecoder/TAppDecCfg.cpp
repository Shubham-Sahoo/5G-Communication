/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2016, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     TAppDecCfg.cpp
    \brief    Decoder configuration class
*/

#include <cstdio>
#include <cstring>
#include <string>
#include "TAppDecCfg.h"
#include "TAppCommon/program_options_lite.h"
#include "TLibCommon/TComChromaFormat.h"
#if SVC_EXTENSION
#include <cassert>
#endif

#ifdef WIN32
#define strdup _strdup
#endif

using namespace std;
namespace po = df::program_options_lite;

//! \ingroup TAppDecoder
//! \{

// ====================================================================================================================
// Public member functions
// ====================================================================================================================

/** \param argc number of arguments
    \param argv array of arguments
 */
Bool TAppDecCfg::parseCfg( Int argc, TChar* argv[] )
{
  Bool do_help = false;
#if SVC_EXTENSION  
  Int layerNum, targetLayerId;
  Int olsIdx;
#if CONFORMANCE_BITSTREAM_MODE
  string confPrefix;
#endif

  Int* cfg_outputBitDepthY[MAX_LAYERS];
  Int* cfg_outputBitDepthC[MAX_LAYERS];

  for( Int layer = 0; layer < MAX_LAYERS; layer++ )
  {
    cfg_outputBitDepthY[layer] = &m_outputBitDepth[layer][CHANNEL_TYPE_LUMA];
    cfg_outputBitDepthC[layer] = &m_outputBitDepth[layer][CHANNEL_TYPE_CHROMA];
  }
#endif

  string cfg_TargetDecLayerIdSetFile;
  string outputColourSpaceConvert;
  Int warnUnknowParameter = 0;

  po::Options opts;
  opts.addOptions()


  ("help",                      do_help,                               false,      "this help text")
  ("BitstreamFile,b",           m_bitstreamFileName,                   string(""), "bitstream input file name")
#if SVC_EXTENSION
  ("c",                         po::parseConfigFile,                               "configuration file name")
  ("ReconFile%d,-o%d",          m_reconFileName,           string(""), MAX_LAYERS, "Layer %d reconstructed YUV output file name\n"
                                                                                   "YUV writing is skipped if omitted")
#if AVC_BASE
  ("BLReconFile,-ibl",                              m_reconFileNameBL, string(""), "BL reconstructed YUV input file name")
#endif
  ("TargetLayerId,-lid",                                        targetLayerId, -1, "Target layer id")
  ("LayerNum,-ls",                                    layerNum, MAX_NUM_LAYER_IDS, "Target layer id") // Legacy option
  ("OutpuLayerSetIdx,-olsidx",                                         olsIdx, -1, "Index of output layer set to be decoded.")
#if CONFORMANCE_BITSTREAM_MODE
  ("ConformanceBitstremMode,-confMode",                      m_confModeFlag, false, "Enable generation of conformance bitstream metadata; True: Generate metadata, False: No metadata generated")
  ("ConformanceMetadataPrefix,-confPrefix",                 confPrefix, string(""), "Prefix for the file name of the conformance data. Default name - 'decodedBitstream'")
#endif
#else
  ("ReconFile,o",               m_reconFileName,                       string(""), "reconstructed YUV output file name\n"
                                                                                   "YUV writing is skipped if omitted")
#endif
  ("WarnUnknowParameter,w",     warnUnknowParameter,                                  0, "warn for unknown configuration parameters instead of failing")
  ("SkipFrames,s",              m_iSkipFrame,                          0,          "number of frames to skip before random access")
#if SVC_EXTENSION
  ("OutputBitDepth%d,%d",       cfg_outputBitDepthY,                0, MAX_LAYERS, "bit depth of YUV output luma component (default: use 0 for native depth)")
  ("OutputBitDepthC%d,%d",      cfg_outputBitDepthC,                0, MAX_LAYERS, "bit depth of YUV output chroma component (default: use 0 for native depth)")
#else
  ("OutputBitDepth,d",          m_outputBitDepth[CHANNEL_TYPE_LUMA],   0,          "bit depth of YUV output luma component (default: use 0 for native depth)")
  ("OutputBitDepthC,d",         m_outputBitDepth[CHANNEL_TYPE_CHROMA], 0,          "bit depth of YUV output chroma component (default: use 0 for native depth)")
#endif
  ("OutputColourSpaceConvert",  outputColourSpaceConvert,              string(""), "Colour space conversion to apply to input 444 video. Permitted values are (empty string=UNCHANGED) " + getListOfColourSpaceConverts(false))
  ("MaxTemporalLayer,t",        m_iMaxTemporalLayer,                   -1,         "Maximum Temporal Layer to be decoded. -1 to decode all layers")
  ("SEIDecodedPictureHash",     m_decodedPictureHashSEIEnabled,        1,          "Control handling of decoded picture hash SEI messages\n"
                                                                                   "\t1: check hash in SEI messages if available in the bitstream\n"
                                                                                   "\t0: ignore SEI message")
  ("SEINoDisplay",              m_decodedNoDisplaySEIEnabled,          true,       "Control handling of decoded no display SEI messages")
  ("TarDecLayerIdSetFile,l",    cfg_TargetDecLayerIdSetFile,           string(""), "targetDecLayerIdSet file name. The file should include white space separated LayerId values to be decoded. Omitting the option or a value of -1 in the file decodes all layers.")
  ("RespectDefDispWindow,w",    m_respectDefDispWindow,                0,          "Only output content inside the default display window\n")
  ("SEIColourRemappingInfoFilename",  m_colourRemapSEIFileName,        string(""), "Colour Remapping YUV output file name. If empty, no remapping is applied (ignore SEI message)\n")
#if O0043_BEST_EFFORT_DECODING
  ("ForceDecodeBitDepth",       m_forceDecodeBitDepth,                 0U,         "Force the decoder to operate at a particular bit-depth (best effort decoding)")
#endif
  ("OutputDecodedSEIMessagesFilename",  m_outputDecodedSEIMessagesFilename,    string(""), "When non empty, output decoded SEI messages to the indicated file. If file is '-', then output to stdout\n")
  ("ClipOutputVideoToRec709Range",      m_bClipOutputVideoToRec709Range,  false, "If true then clip output video to the Rec. 709 Range on saving")
  ;

  po::setDefaults(opts);
  po::ErrorReporter err;
  const list<const TChar*>& argv_unhandled = po::scanArgv(opts, argc, (const TChar**) argv, err);

  for (list<const TChar*>::const_iterator it = argv_unhandled.begin(); it != argv_unhandled.end(); it++)
  {
    fprintf(stderr, "Unhandled argument ignored: `%s'\n", *it);
  }

  if (argc == 1 || do_help)
  {
    po::doHelp(cout, opts);
    return false;
  }

  if (err.is_errored)
  {
    if (!warnUnknowParameter)
    {
      /* errors have already been reported to stderr */
      return false;
    }
  }

  m_outputColourSpaceConvert = stringToInputColourSpaceConvert(outputColourSpaceConvert, false);
  if (m_outputColourSpaceConvert>=NUMBER_INPUT_COLOUR_SPACE_CONVERSIONS)
  {
    fprintf(stderr, "Bad output colour space conversion string\n");
    return false;
  }

#if SVC_EXTENSION
  if( targetLayerId < 0 )
  {
    targetLayerId = MAX_VPS_LAYER_IDX_PLUS1 - 1;
  }

  assert( targetLayerId >= 0 );
  assert( targetLayerId < MAX_NUM_LAYER_IDS );

#if CONFORMANCE_BITSTREAM_MODE
  if( m_confModeFlag )
  {
    assert( olsIdx != -1 ); // In the conformance mode, target output layer set index is to be explicitly specified.

    if( confPrefix.empty() )
    {
      m_confPrefix = string("decodedBitstream");
    }
    else
    {
      m_confPrefix = confPrefix;
    }
      // Open metadata file and write
    char fileNameSuffix[255];
    sprintf(fileNameSuffix, "%s-OLS%d.opl", m_confPrefix.c_str(), olsIdx);  // olsIdx is the target output layer set index.
    m_metadataFileName = string(fileNameSuffix);
    m_metadataFileRefresh = true;

    // Decoded layer YUV files
    for(Int layer= 0; layer < MAX_VPS_LAYER_IDX_PLUS1; layer++ )
    {
      sprintf(fileNameSuffix, "%s-L%d.yuv", m_confPrefix.c_str(), layer);  // olsIdx is the target output layer set index.
      m_decodedYuvLayerFileName[layer] = std::string( fileNameSuffix );
      m_decodedYuvLayerRefresh[layer] = true;
    }
  }
#endif
  m_commonDecoderParams.setTargetOutputLayerSetIdx( olsIdx );
  m_commonDecoderParams.setTargetLayerId( targetLayerId );
#endif

  if (m_bitstreamFileName.empty())
  {
    fprintf(stderr, "No input file specified, aborting\n");
    return false;
  }

  if ( !cfg_TargetDecLayerIdSetFile.empty() )
  {
    FILE* targetDecLayerIdSetFile = fopen ( cfg_TargetDecLayerIdSetFile.c_str(), "r" );
    if ( targetDecLayerIdSetFile )
    {
      Bool isLayerIdZeroIncluded = false;
      while ( !feof(targetDecLayerIdSetFile) )
      {
        Int layerIdParsed = 0;
        if ( fscanf( targetDecLayerIdSetFile, "%d ", &layerIdParsed ) != 1 )
        {
          if ( m_targetDecLayerIdSet.size() == 0 )
          {
            fprintf(stderr, "No LayerId could be parsed in file %s. Decoding all LayerIds as default.\n", cfg_TargetDecLayerIdSetFile.c_str() );
          }
          break;
        }
        if ( layerIdParsed  == -1 ) // The file includes a -1, which means all LayerIds are to be decoded.
        {
          m_targetDecLayerIdSet.clear(); // Empty set means decoding all layers.
          break;
        }
        if ( layerIdParsed < 0 || layerIdParsed >= MAX_NUM_LAYER_IDS )
        {
          fprintf(stderr, "Warning! Parsed LayerId %d is not within allowed range [0,%d]. Ignoring this value.\n", layerIdParsed, MAX_NUM_LAYER_IDS-1 );
        }
        else
        {
          isLayerIdZeroIncluded = layerIdParsed == 0 ? true : isLayerIdZeroIncluded;
          m_targetDecLayerIdSet.push_back ( layerIdParsed );
        }
      }
      fclose (targetDecLayerIdSetFile);
      if ( m_targetDecLayerIdSet.size() > 0 && !isLayerIdZeroIncluded )
      {
        fprintf(stderr, "TargetDecLayerIdSet must contain LayerId=0, aborting" );
        return false;
      }
    }
    else
    {
      fprintf(stderr, "File %s could not be opened. Using all LayerIds as default.\n", cfg_TargetDecLayerIdSetFile.c_str() );
    }

#if SVC_EXTENSION
    m_commonDecoderParams.setTargetDecLayerIdSet( &m_targetDecLayerIdSet );
#endif
  }

  return true;
}

//! \}
