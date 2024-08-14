use std::io;
use std::time::Duration;

use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, MouseEventKind};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use num_traits::AsPrimitive;
use ratatui::prelude::Rect;
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::Terminal;
use ratatui::widgets::{Block, Borders, Chart, Dataset, GraphType, Paragraph, Wrap};

use rand::NoiseVec;
use statrs::statistics::Statistics;

fn main() -> Result<(), io::Error> {
    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let count = 1_000_000;

    let app = App::new(count);
    let res = run_app(&mut terminal, app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    Ok(())
}

struct App {
    count: usize,
    noise: Vec<f64>,
}

#[allow(dead_code)]
impl App {
    fn new(count: impl AsPrimitive<usize>) -> Self {
        let noise = Vec::<f64>::with_noise(count);
        Self { count: count.as_(), noise }
    }

    fn line(count: impl AsPrimitive<usize>, slope: impl AsPrimitive<f64>, intercept: impl AsPrimitive<f64>) -> Self {
        let mut noise = Vec::<f64>::with_capacity(count.as_());
        for i in 0..count.as_() {
            noise.push(slope.as_() * i as f64 + intercept.as_());
        }
        Self { count: count.as_(), noise }
    }

    fn regenerate_noise(&mut self) {
        self.noise = Vec::<f64>::with_noise(self.count);
    }
}

fn run_app<B: Backend>(terminal: &mut Terminal<B>, mut app: App) -> io::Result<()> {
    let mut mouse_position: Option<(u16, u16)> = None;

    loop {
        terminal.draw(|f| {
            let size = f.area();
            let terminal_width = size.width as f64;
            let terminal_height = size.height as f64;
            let scaled_noise = calculate_scaled_noise(&app.noise, terminal_width, terminal_height);

            let mean = (&app.noise).mean();
            let std_dev = (&app.noise).std_dev();

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(80), Constraint::Percentage(20)].as_ref())
                .split(size);

            render_chart(f, &chunks, terminal_width, terminal_height, mouse_position, &scaled_noise, mean, std_dev);
            render_tooltip(f, mouse_position, &chunks, terminal_width, terminal_height, &scaled_noise);
        })?;

        if handle_event(terminal, &mut app, &mut mouse_position)? {
            return Ok(());
        }
    }
}

fn build_chart(datasets: Vec<Dataset>, terminal_width: f64, y_min: f64, y_max: f64) -> Chart {
    let x_axis = ratatui::widgets::Axis::default()
        .bounds([0.0, terminal_width])
        .labels(vec![
            format!("{:.0}", 0.0),
            format!("{:.0}", terminal_width / 2.0),
            format!("{:.0}", terminal_width),
        ])
        .style(Style::default().fg(Color::Gray));

    let y_axis = ratatui::widgets::Axis::default()
        .bounds([y_min, y_max])
        .labels(vec![
            format!("{:.2}", y_min),
            format!("{:.2}", (y_min + y_max) / 2.0),
            format!("{:.2}", y_max),
        ])
        .style(Style::default().fg(Color::Gray));

    Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title("1D Noise Oscilloscope"))
        .x_axis(x_axis)
        .y_axis(y_axis)
}

fn build_stats(size: Rect, mean: f64, std_dev: f64, points: usize) -> Paragraph<'static> {
    Paragraph::new(format!(
        "Size: {size} | Mean: {mean:.2} | Std Dev: {std_dev:.2} | Points: {points}"
    ))
    .block(Block::default().borders(Borders::ALL).title("Scale Info"))
    .wrap(Wrap { trim: true })
}

fn build_tooltip(x_index: usize, y_value: f64) -> Paragraph<'static> {
    Paragraph::new(format!("X: {} | Y: {:.2}", x_index, y_value))
        .block(Block::default().borders(Borders::NONE))
        .wrap(Wrap { trim: true })
        .alignment(ratatui::layout::Alignment::Left)
        .style(Style::default())
}

fn calculate_scaled_noise(noise: &[f64], terminal_width: f64, terminal_height: f64) -> Vec<(f64, f64)> {
    // // let x_max = noise.len() as f64 - 1.0;
    // let y_min = noise.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    // let y_max = noise.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // let x: Vec<(f64, f64)> = noise.iter()
    //     .enumerate()
    //     .filter(|(x, _)| *x < terminal_width.as_())
    //     .map(|(x, y)| {
    //         let normalized_y = (y - y_min) / (y_max - y_min);
    //         let scaled_y = (1.0 - normalized_y) * (terminal_height - 1.0);
    //         (x.as_(), scaled_y)
    //     })
    //     .collect();
    // assert!(x.len() == terminal_width.as_());
    noise.iter()
    .enumerate()
    .filter(|(x, _)| *x < terminal_width.as_())
    .map(|(x, y)| {
        let scaled_y = (1.0 - y) * (terminal_height - 1.0);
        (x.as_(), scaled_y)
    }).collect()

}

#[allow(unused_variables)]
fn handle_event<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    mouse_position: &mut Option<(u16, u16)>
) -> io::Result<bool> {
    if crossterm::event::poll(Duration::from_millis(10))? {
        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return Ok(true),
                KeyCode::Char(' ') => app.regenerate_noise(),
                _ => {}
            }
        } else if let Event::Mouse(mouse_event) = event::read()? {
            if let MouseEventKind::Moved = mouse_event.kind {
                *mouse_position = Some((mouse_event.column, mouse_event.row));
            }
        }
    }
    Ok(false)
}

fn render_chart(
    f: &mut ratatui::Frame,
    chunks: &[Rect],
    terminal_width: f64,
    _terminal_height: f64,
    mouse_position: Option<(u16, u16)>,
    scaled_noise: &[(f64, f64)],
    noise_mean: f64,
    noise_std_dev: f64,
) {
    let (y_min, y_max) = scaled_noise.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), &(_, y)| {
        (min_y.min(y), max_y.max(y))
    });

    let mut datasets: Vec<Dataset> = vec![
        Dataset::default()
            .name("Noise")
            .marker(ratatui::symbols::Marker::Braille)
            .style(Style::default().fg(Color::Cyan))
            .graph_type(GraphType::Line)
            .data(scaled_noise),
    ];

    let mut highlight_line = vec![];
    if let Some((mouse_x, _)) = mouse_position {
        let x_index = ((mouse_x as f64 / terminal_width) * scaled_noise.len() as f64).round() as usize;
        if x_index < scaled_noise.len() {
            let x_pos = scaled_noise[x_index].0;
            highlight_line.push((x_pos, y_min));
            highlight_line.push((x_pos, y_max));
        }
    }
    datasets.push(
        Dataset::default()
            .name("Highlight")
            .marker(ratatui::symbols::Marker::Braille)
            .style(Style::default().fg(Color::Red))
            .graph_type(GraphType::Line)
            .data(&highlight_line),
    );

    let chart = build_chart(datasets, terminal_width, y_min, y_max);
    let stats = build_stats(
        chunks[0],
        noise_mean,
        noise_std_dev,
        scaled_noise.len(),
    );

    f.render_widget(chart, chunks[0]);
    f.render_widget(stats, chunks[1]);
}

fn render_tooltip(
    f: &mut ratatui::Frame,
    mouse_position: Option<(u16, u16)>,
    chunks: &[Rect],
    terminal_width: f64,
    terminal_height: f64,
    scaled_noise: &[(f64, f64)],
) -> Option<()> {
    if let Some((mouse_x, mouse_y)) = mouse_position {
        // Check if the mouse is outside the terminal boundaries
        if mouse_x >= terminal_width as u16 || mouse_y >= terminal_height as u16 {
            return None; // Hide the tooltip if the mouse is outside the terminal
        }

        let chart_area = chunks[0];
        if mouse_x >= chart_area.x
            && mouse_x < chart_area.x + chart_area.width
            && mouse_y >= chart_area.y
            && mouse_y < chart_area.y + chart_area.height
        {
            let x_index = ((mouse_x as f64 / terminal_width) * scaled_noise.len() as f64).round() as usize;
            if x_index < scaled_noise.len() {
                let y_value = scaled_noise[x_index].1;

                // Adjust tooltip position to stay within terminal bounds
                let tooltip_width = 20;
                let tooltip_height = 3;

                let tooltip_x = if mouse_x + tooltip_width > terminal_width as u16 {
                    terminal_width as u16 - tooltip_width
                } else {
                    mouse_x
                };

                let tooltip_y = if mouse_y + tooltip_height > terminal_height as u16 {
                    terminal_height as u16 - tooltip_height
                } else {
                    mouse_y
                };

                let tooltip = build_tooltip(x_index, y_value);
                f.render_widget(tooltip, Rect::new(tooltip_x, tooltip_y, tooltip_width, tooltip_height));
            }
        } else {
            return None; // Mouse is out of the chart bounds
        }
    }
    Some(())
}
