/*
 * Copyright (c) 2001 Heikki Leinonen
 * Copyright (c) 2001 Chris Bagwell
 * Copyright (c) 2003 Donnie Smith
 * Copyright (c) 2014 Paul B Mahol
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <float.h> /* DBL_MAX */

#include "libavutil/opt.h"
#include "libavutil/timestamp.h"
#include "libavutil/audio_fifo.h"
#include "audio.h"
#include "formats.h"
#include "avfilter.h"
#include "internal.h"

enum SilenceMode {
    SILENCE_TRIM,
    SILENCE_TRIM_FLUSH,
    SILENCE_COPY,
    SILENCE_COPY_FLUSH,
    SILENCE_STOP,
    SILENCETRIM_START,
    SILENCETRIM_COPY,
    SILENCETRIM_BUFFER,
};

typedef struct SilenceRemoveContext {
    const AVClass *class;

    enum SilenceMode mode;

    int start_periods;
    int64_t start_duration;
    double start_threshold;

    int stop_periods;
    int64_t stop_duration;
    double stop_threshold;

    double *start_holdoff;
    size_t start_holdoff_offset;
    size_t start_holdoff_end;
    int    start_found_periods;

    double *stop_holdoff;
    size_t stop_holdoff_offset;
    size_t stop_holdoff_end;
    int    stop_found_periods;

    double window_ratio;
    double *window;
    double *window_current;
    double *window_end;
    int window_size;
    double sum;

    int leave_silence;
    int restart;
    int64_t next_pts;

    int detection;
    void (*update)(struct SilenceRemoveContext *s, double sample);
    double(*compute)(struct SilenceRemoveContext *s, double sample);

    AVAudioFifo *fifo;
} SilenceRemoveContext;

#define OFFSET(x) offsetof(SilenceRemoveContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_AUDIO_PARAM
static const AVOption silenceremove_options[] = {
    { "start_periods",   NULL, OFFSET(start_periods),   AV_OPT_TYPE_INT,      {.i64=0},     0,    9000, FLAGS },
    { "start_duration",  NULL, OFFSET(start_duration),  AV_OPT_TYPE_DURATION, {.i64=0},     0,    9000, FLAGS },
    { "start_threshold", NULL, OFFSET(start_threshold), AV_OPT_TYPE_DOUBLE,   {.dbl=0},     0, DBL_MAX, FLAGS },
    { "stop_periods",    NULL, OFFSET(stop_periods),    AV_OPT_TYPE_INT,      {.i64=0}, -9000,    9000, FLAGS },
    { "stop_duration",   NULL, OFFSET(stop_duration),   AV_OPT_TYPE_DURATION, {.i64=0},     0,    9000, FLAGS },
    { "stop_threshold",  NULL, OFFSET(stop_threshold),  AV_OPT_TYPE_DOUBLE,   {.dbl=0},     0, DBL_MAX, FLAGS },
    { "leave_silence",   NULL, OFFSET(leave_silence),   AV_OPT_TYPE_BOOL,     {.i64=0},     0,       1, FLAGS },
    { "detection",       NULL, OFFSET(detection),       AV_OPT_TYPE_INT,      {.i64=1},     0,       1, FLAGS, "detection" },
    {   "peak",          0,    0,                       AV_OPT_TYPE_CONST,    {.i64=0},     0,       0, FLAGS, "detection" },
    {   "rms",           0,    0,                       AV_OPT_TYPE_CONST,    {.i64=1},     0,       0, FLAGS, "detection" },
    { "window",          NULL, OFFSET(window_ratio),    AV_OPT_TYPE_DOUBLE,   {.dbl=0.02},  0,      10, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(silenceremove);

static double compute_peak(SilenceRemoveContext *s, double sample)
{
    double new_sum;

    new_sum  = s->sum;
    new_sum -= *s->window_current;
    new_sum += fabs(sample);

    return new_sum / s->window_size;
}

static void update_peak(SilenceRemoveContext *s, double sample)
{
    s->sum -= *s->window_current;
    *s->window_current = fabs(sample);
    s->sum += *s->window_current;

    s->window_current++;
    if (s->window_current >= s->window_end)
        s->window_current = s->window;
}

static double compute_rms(SilenceRemoveContext *s, double sample)
{
    double new_sum;

    new_sum  = s->sum;
    new_sum -= *s->window_current;
    new_sum += sample * sample;

    return sqrt(new_sum / s->window_size);
}

static void update_rms(SilenceRemoveContext *s, double sample)
{
    s->sum -= *s->window_current;
    *s->window_current = sample * sample;
    s->sum += *s->window_current;

    s->window_current++;
    if (s->window_current >= s->window_end)
        s->window_current = s->window;
}

static av_cold int init(AVFilterContext *ctx)
{
    SilenceRemoveContext *s = ctx->priv;

    if (s->stop_periods < 0) {
        s->stop_periods = -s->stop_periods;
        s->restart = 1;
    }

    switch (s->detection) {
    case 0:
        s->update = update_peak;
        s->compute = compute_peak;
        break;
    case 1:
        s->update = update_rms;
        s->compute = compute_rms;
        break;
    };

    return 0;
}

static void clear_window(SilenceRemoveContext *s)
{
    memset(s->window, 0, s->window_size * sizeof(*s->window));

    s->window_current = s->window;
    s->window_end = s->window + s->window_size;
    s->sum = 0;
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    SilenceRemoveContext *s = ctx->priv;

    s->window_size = FFMAX((inlink->sample_rate * s->window_ratio), 1) * inlink->channels;
    s->window = av_malloc_array(s->window_size, sizeof(*s->window));
    if (!s->window)
        return AVERROR(ENOMEM);

    clear_window(s);

    s->start_duration = av_rescale(s->start_duration, inlink->sample_rate,
                                   AV_TIME_BASE);
    s->stop_duration  = av_rescale(s->stop_duration, inlink->sample_rate,
                                   AV_TIME_BASE);

    s->start_holdoff = av_malloc_array(FFMAX(s->start_duration, 1),
                                       sizeof(*s->start_holdoff) *
                                       inlink->channels);
    if (!s->start_holdoff)
        return AVERROR(ENOMEM);

    s->start_holdoff_offset = 0;
    s->start_holdoff_end    = 0;
    s->start_found_periods  = 0;

    s->stop_holdoff = av_malloc_array(FFMAX(s->stop_duration, 1),
                                      sizeof(*s->stop_holdoff) *
                                      inlink->channels);
    if (!s->stop_holdoff)
        return AVERROR(ENOMEM);

    s->stop_holdoff_offset = 0;
    s->stop_holdoff_end    = 0;
    s->stop_found_periods  = 0;

    if (s->start_periods)
        s->mode = SILENCE_TRIM;
    else
        s->mode = SILENCE_COPY;

    return 0;
}

static void flush(SilenceRemoveContext *s,
                  AVFrame *out, AVFilterLink *outlink,
                  int *nb_samples_written, int *ret)
{
    if (*nb_samples_written) {
        out->nb_samples = *nb_samples_written / outlink->channels;

        out->pts = s->next_pts;
        s->next_pts += av_rescale_q(out->nb_samples,
                                    (AVRational){1, outlink->sample_rate},
                                    outlink->time_base);

        *ret = ff_filter_frame(outlink, out);
        *nb_samples_written = 0;
    } else {
        av_frame_free(&out);
    }
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    SilenceRemoveContext *s = ctx->priv;
    int i, j, threshold, ret = 0;
    int nbs, nb_samples_read, nb_samples_written;
    double *obuf, *ibuf = (double *)in->data[0];
    AVFrame *out;

    nb_samples_read = nb_samples_written = 0;

    switch (s->mode) {
    case SILENCE_TRIM:
silence_trim:
        nbs = in->nb_samples - nb_samples_read / inlink->channels;
        if (!nbs)
            break;

        for (i = 0; i < nbs; i++) {
            threshold = 0;
            for (j = 0; j < inlink->channels; j++) {
                threshold |= s->compute(s, ibuf[j]) > s->start_threshold;
            }

            if (threshold) {
                for (j = 0; j < inlink->channels; j++) {
                    s->update(s, *ibuf);
                    s->start_holdoff[s->start_holdoff_end++] = *ibuf++;
                }
                nb_samples_read += inlink->channels;

                if (s->start_holdoff_end >= s->start_duration * inlink->channels) {
                    if (++s->start_found_periods >= s->start_periods) {
                        s->mode = SILENCE_TRIM_FLUSH;
                        goto silence_trim_flush;
                    }

                    s->start_holdoff_offset = 0;
                    s->start_holdoff_end = 0;
                }
            } else {
                s->start_holdoff_end = 0;

                for (j = 0; j < inlink->channels; j++)
                    s->update(s, ibuf[j]);

                ibuf += inlink->channels;
                nb_samples_read += inlink->channels;
            }
        }
        break;

    case SILENCE_TRIM_FLUSH:
silence_trim_flush:
        nbs  = s->start_holdoff_end - s->start_holdoff_offset;
        nbs -= nbs % inlink->channels;
        if (!nbs)
            break;

        out = ff_get_audio_buffer(inlink, nbs / inlink->channels);
        if (!out) {
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }

        memcpy(out->data[0], &s->start_holdoff[s->start_holdoff_offset],
               nbs * sizeof(double));

        out->pts = s->next_pts;
        s->next_pts += av_rescale_q(out->nb_samples,
                                    (AVRational){1, outlink->sample_rate},
                                    outlink->time_base);

        s->start_holdoff_offset += nbs;

        ret = ff_filter_frame(outlink, out);

        if (s->start_holdoff_offset == s->start_holdoff_end) {
            s->start_holdoff_offset = 0;
            s->start_holdoff_end = 0;
            s->mode = SILENCE_COPY;
            goto silence_copy;
        }
        break;

    case SILENCE_COPY:
silence_copy:
        nbs = in->nb_samples - nb_samples_read / inlink->channels;
        if (!nbs)
            break;

        out = ff_get_audio_buffer(inlink, nbs);
        if (!out) {
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }
        obuf = (double *)out->data[0];

        if (s->stop_periods) {
            for (i = 0; i < nbs; i++) {
                threshold = 1;
                for (j = 0; j < inlink->channels; j++)
                    threshold &= s->compute(s, ibuf[j]) > s->stop_threshold;

                if (threshold && s->stop_holdoff_end && !s->leave_silence) {
                    s->mode = SILENCE_COPY_FLUSH;
                    flush(s, out, outlink, &nb_samples_written, &ret);
                    goto silence_copy_flush;
                } else if (threshold) {
                    for (j = 0; j < inlink->channels; j++) {
                        s->update(s, *ibuf);
                        *obuf++ = *ibuf++;
                    }
                    nb_samples_read    += inlink->channels;
                    nb_samples_written += inlink->channels;
                } else if (!threshold) {
                    for (j = 0; j < inlink->channels; j++) {
                        s->update(s, *ibuf);
                        if (s->leave_silence) {
                            *obuf++ = *ibuf;
                            nb_samples_written++;
                        }

                        s->stop_holdoff[s->stop_holdoff_end++] = *ibuf++;
                    }
                    nb_samples_read += inlink->channels;

                    if (s->stop_holdoff_end >= s->stop_duration * inlink->channels) {
                        if (++s->stop_found_periods >= s->stop_periods) {
                            s->stop_holdoff_offset = 0;
                            s->stop_holdoff_end = 0;

                            if (!s->restart) {
                                s->mode = SILENCE_STOP;
                                flush(s, out, outlink, &nb_samples_written, &ret);
                                goto silence_stop;
                            } else {
                                s->stop_found_periods = 0;
                                s->start_found_periods = 0;
                                s->start_holdoff_offset = 0;
                                s->start_holdoff_end = 0;
                                clear_window(s);
                                s->mode = SILENCE_TRIM;
                                flush(s, out, outlink, &nb_samples_written, &ret);
                                goto silence_trim;
                            }
                        }
                        s->mode = SILENCE_COPY_FLUSH;
                        flush(s, out, outlink, &nb_samples_written, &ret);
                        goto silence_copy_flush;
                    }
                }
            }
            flush(s, out, outlink, &nb_samples_written, &ret);
        } else {
            memcpy(obuf, ibuf, sizeof(double) * nbs * inlink->channels);

            out->pts = s->next_pts;
            s->next_pts += av_rescale_q(out->nb_samples,
                                        (AVRational){1, outlink->sample_rate},
                                        outlink->time_base);

            ret = ff_filter_frame(outlink, out);
        }
        break;

    case SILENCE_COPY_FLUSH:
silence_copy_flush:
        nbs  = s->stop_holdoff_end - s->stop_holdoff_offset;
        nbs -= nbs % inlink->channels;
        if (!nbs)
            break;

        out = ff_get_audio_buffer(inlink, nbs / inlink->channels);
        if (!out) {
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }

        memcpy(out->data[0], &s->stop_holdoff[s->stop_holdoff_offset],
               nbs * sizeof(double));
        s->stop_holdoff_offset += nbs;

        out->pts = s->next_pts;
        s->next_pts += av_rescale_q(out->nb_samples,
                                    (AVRational){1, outlink->sample_rate},
                                    outlink->time_base);

        ret = ff_filter_frame(outlink, out);

        if (s->stop_holdoff_offset == s->stop_holdoff_end) {
            s->stop_holdoff_offset = 0;
            s->stop_holdoff_end = 0;
            s->mode = SILENCE_COPY;
            goto silence_copy;
        }
        break;
    case SILENCE_STOP:
silence_stop:
        break;
    }

    av_frame_free(&in);

    return ret;
}

static int request_frame(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    SilenceRemoveContext *s = ctx->priv;
    int ret;

    ret = ff_request_frame(ctx->inputs[0]);
    if (ret == AVERROR_EOF && (s->mode == SILENCE_COPY_FLUSH ||
                               s->mode == SILENCE_COPY)) {
        int nbs = s->stop_holdoff_end - s->stop_holdoff_offset;
        if (nbs) {
            AVFrame *frame;

            frame = ff_get_audio_buffer(outlink, nbs / outlink->channels);
            if (!frame)
                return AVERROR(ENOMEM);

            memcpy(frame->data[0], &s->stop_holdoff[s->stop_holdoff_offset],
                   nbs * sizeof(double));

            frame->pts = s->next_pts;
            s->next_pts += av_rescale_q(frame->nb_samples,
                                        (AVRational){1, outlink->sample_rate},
                                        outlink->time_base);

            ret = ff_filter_frame(ctx->inputs[0], frame);
        }
        s->mode = SILENCE_STOP;
    }
    return ret;
}

static int query_formats(AVFilterContext *ctx)
{
    AVFilterFormats *formats = NULL;
    AVFilterChannelLayouts *layouts = NULL;
    static const enum AVSampleFormat sample_fmts[] = {
        AV_SAMPLE_FMT_DBL, AV_SAMPLE_FMT_NONE
    };
    int ret;

    layouts = ff_all_channel_counts();
    if (!layouts)
        return AVERROR(ENOMEM);
    ret = ff_set_common_channel_layouts(ctx, layouts);
    if (ret < 0)
        return ret;

    formats = ff_make_format_list(sample_fmts);
    if (!formats)
        return AVERROR(ENOMEM);
    ret = ff_set_common_formats(ctx, formats);
    if (ret < 0)
        return ret;

    formats = ff_all_samplerates();
    if (!formats)
        return AVERROR(ENOMEM);
    return ff_set_common_samplerates(ctx, formats);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    SilenceRemoveContext *s = ctx->priv;

    av_freep(&s->start_holdoff);
    av_freep(&s->stop_holdoff);
    av_freep(&s->window);
}

static const AVFilterPad silenceremove_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_AUDIO,
        .config_props = config_input,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad silenceremove_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_AUDIO,
        .request_frame = request_frame,
    },
    { NULL }
};

AVFilter ff_af_silenceremove = {
    .name          = "silenceremove",
    .description   = NULL_IF_CONFIG_SMALL("Remove silence."),
    .priv_size     = sizeof(SilenceRemoveContext),
    .priv_class    = &silenceremove_class,
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = silenceremove_inputs,
    .outputs       = silenceremove_outputs,
};




static const AVOption silencetrim_options[] = {
    { "start_threshold", NULL, OFFSET(start_threshold), AV_OPT_TYPE_DOUBLE,   {.dbl=0},     0, DBL_MAX, FLAGS },
    { "stop_threshold",  NULL, OFFSET(stop_threshold),  AV_OPT_TYPE_DOUBLE,   {.dbl=0},     0, DBL_MAX, FLAGS },
    { "detection",       NULL, OFFSET(detection),       AV_OPT_TYPE_INT,      {.i64=1},     0,       1, FLAGS, "detection" },
    {   "peak",          0,    0,                       AV_OPT_TYPE_CONST,    {.i64=0},     0,       0, FLAGS, "detection" },
    {   "rms",           0,    0,                       AV_OPT_TYPE_CONST,    {.i64=1},     0,       0, FLAGS, "detection" },
    { "window",          NULL, OFFSET(window_ratio),    AV_OPT_TYPE_DOUBLE,   {.dbl=0.02},  0,      10, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(silencetrim);


static int trim_config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    SilenceRemoveContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];

    s->window_size = FFMAX((inlink->sample_rate * s->window_ratio), 1) * inlink->channels;
    s->window = av_malloc_array(s->window_size, sizeof(*s->window));
    if (!s->window)
        return AVERROR(ENOMEM);

    clear_window(s);

    s->mode = SILENCETRIM_START;
    s->fifo = av_audio_fifo_alloc(outlink->format, outlink->channels, inlink->sample_rate * 4);
    if (!s->fifo) {
        return AVERROR(ENOMEM);
    }

    return 0;
}

static int filter_subframe(AVFilterLink *inlink, AVFrame *in, int start_index, int end_index) {
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    SilenceRemoveContext *s = ctx->priv;
    AVFrame *out = ff_get_audio_buffer(outlink, end_index - start_index + 1);
    if (!out) {
        return AVERROR(ENOMEM);
    }

    memcpy(out->data[0], in->data[0] + start_index * inlink->channels * sizeof(double), out->nb_samples * inlink->channels * sizeof(double));
    out->pts = s->next_pts;
    s->next_pts += av_rescale_q(out->nb_samples,
                    (AVRational){1, outlink->sample_rate},
                    outlink->time_base);
    return ff_filter_frame(outlink, out);
}

static void buffer_frame_end(AVFilterLink *inlink, AVFrame *in, int start_index) {
    AVFilterContext *ctx = inlink->dst;
    SilenceRemoveContext *s = ctx->priv;
    int out_samples = in->nb_samples - start_index;
    double *sub_frame_data = (double*)in->data[0] + start_index * inlink ->channels;
    av_audio_fifo_write(s->fifo, (void**)&sub_frame_data, out_samples);
}

static int trim_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    SilenceRemoveContext *s = ctx->priv;
    double *ibuf = (double *)in->data[0];
    int i, j;
    int above_start_threshold;
    int above_stop_threshold;
    double sample_volume;
    int first_non_silence_sample_in_frame = -1;
    int last_non_silence_sample_in_frame = -1;
    int ret;
    AVFrame *out;

    for (i = 0;i < in->nb_samples; i++) {
        above_start_threshold = 0;
        above_stop_threshold = 0;
        for (j = 0; j < inlink->channels; j++) {
            sample_volume = s->compute(s, ibuf[i * inlink->channels + j]);
            //sample_volume might be NaN, but we want to get a false for this case anyway. (So NaN = assume silence)
            above_start_threshold |= sample_volume > s->start_threshold;
            above_stop_threshold  |= sample_volume > s->stop_threshold;
        }
        if (above_start_threshold && first_non_silence_sample_in_frame == -1) {
            first_non_silence_sample_in_frame = i;
            last_non_silence_sample_in_frame = i;
        }
        if (above_stop_threshold) {
            last_non_silence_sample_in_frame = i;
        }
        for (j = 0; j < inlink->channels; j++) {
            s->update(s, ibuf[i * inlink->channels + j]);
        }

    }
    if (s->mode == SILENCETRIM_START) {
        //We were trimming silence from the start of the audio stream
        if (first_non_silence_sample_in_frame != -1) {
            //The audio started playing in this frame
            //At this point we discard the data that was used to calculate the window.
            ret = filter_subframe(inlink, in, first_non_silence_sample_in_frame, last_non_silence_sample_in_frame);
            if (ret < 0) {
                return ret;
            }

            if (last_non_silence_sample_in_frame < in->nb_samples - 1) {
                //Silence also started in this frame: Buffer remaining samples
                buffer_frame_end(inlink, in, last_non_silence_sample_in_frame + 1);

                s->mode = SILENCETRIM_BUFFER;
            } else {
                s->mode = SILENCETRIM_COPY;
            }
        }
    } else if (s->mode == SILENCETRIM_COPY) {
        //We are currently copying data, so we need to generate a frame with all unwritten data.
        //Generate Frame from 0 to last_non_silence_sample_in_frame(Might be the whole frame)
        assert(last_non_silence_sample_in_frame >= 0);
        ret = filter_subframe(inlink, in, 0, last_non_silence_sample_in_frame);
        if (ret < 0) {
            return ret;
        }
        if (last_non_silence_sample_in_frame < in->nb_samples - 1) {
            //Silence also started in this frame: Buffer remaining samples
            buffer_frame_end(inlink, in, last_non_silence_sample_in_frame + 1);

            s->mode = SILENCETRIM_BUFFER;
        }
    } else if (s->mode == SILENCETRIM_BUFFER) {
        //We are currently buffering silence, that might be the end of all audio.
        if (first_non_silence_sample_in_frame != -1) {
            //Silence ended in this frame, so buffered silence should not be trimmed.
            //Generate Frame from buffer + frame from 0 to last_non_silence_sample_in_frame
            if (av_audio_fifo_size(s->fifo) > 0) {
                out = ff_get_audio_buffer(outlink, av_audio_fifo_size(s->fifo));
                if (!out) {
                    return AVERROR(ENOMEM);
                }
                av_audio_fifo_read(s->fifo, (void**)out->extended_data, out->nb_samples);

                out->pts = s->next_pts;
                s->next_pts += av_rescale_q(out->nb_samples,
                    (AVRational){1, outlink->sample_rate},
                    outlink->time_base);
                ret = ff_filter_frame(outlink, out);
                if (ret < 0) {
                    return ret;
                }
            }
            ret = filter_subframe(inlink, in, 0, last_non_silence_sample_in_frame);
            if (ret < 0) {
                return ret;
            }

            if (last_non_silence_sample_in_frame < in->nb_samples - 1) {
                buffer_frame_end(inlink, in, last_non_silence_sample_in_frame + 1);

                s->mode = SILENCETRIM_BUFFER;
            } else {
                s->mode = SILENCETRIM_COPY;
            }
        } else {
            //Silence did not end in this frame: Buffer the whole frame
            av_audio_fifo_write(s->fifo, (void**)in->data, in->nb_samples);
        }
    }

    av_frame_free(&in);
    return 0;
}

static av_cold void trim_uninit(AVFilterContext *ctx)
{
    SilenceRemoveContext *s = ctx->priv;
    av_audio_fifo_free(s->fifo);
}

static const AVFilterPad silencetrim_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_AUDIO,
        .config_props = trim_config_input,
        .filter_frame = trim_filter_frame,
    },
    { NULL }
};

static const AVFilterPad silencetrim_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_AUDIO,
    },
    { NULL }
};

AVFilter ff_af_silencetrim = {
    .name          = "silencetrim",
    .description   = NULL_IF_CONFIG_SMALL("Trim silence from start and end of audio."),
    .priv_size     = sizeof(SilenceRemoveContext),
    .priv_class    = &silenceremove_class,
    .init          = init,
    .uninit        = trim_uninit,
    .query_formats = query_formats,
    .inputs        = silencetrim_inputs,
    .outputs       = silencetrim_outputs,
};
